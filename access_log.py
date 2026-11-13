"""Email OTP check and a local Excel record for every video translation.

Flow: the person enters name + email + a short description -> we email a 6-digit code ->
they enter it -> ONE translation is allowed -> a row is written to records/translation_log.xlsx.

Sending needs a Gmail account with an App Password, put in the .env file (never committed):
    GMAIL_ADDRESS=you@gmail.com
    GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx
Codes are never stored on disk and never written to the Excel file.
"""
import base64, csv, hashlib, hmac, io, json, os, re, secrets, smtplib, ssl, struct, subprocess, threading, time
from datetime import datetime
from urllib.parse import quote
from email.message import EmailMessage

ROOT = os.path.dirname(os.path.abspath(__file__))
RECORDS_DIR = os.path.join(ROOT, "records")
XLSX = os.path.join(RECORDS_DIR, "translation_log.xlsx")
PENDING_CSV = os.path.join(RECORDS_DIR, "translation_log_pending.csv")

OTP_TTL = 600            # a code is valid for 10 minutes
MAX_TRIES = 5            # wrong guesses before the code is cancelled
RESEND_WAIT = 30         # seconds between "send code" clicks
MAX_SENDS_PER_HOUR = 5   # per email address
VERIFIED_TTL = 1800      # after verifying, you have 30 minutes to start the translation

USERS = os.path.join(RECORDS_DIR, "authenticator_users.json")   # email -> secret (local only)
ISSUER = "Whisper Video Translator"
TOTP_STEP = 30
LOCK_AFTER, LOCK_SECS = 5, 300   # 5 wrong codes for one email -> locked for 5 minutes
_fails = {}                      # email -> [timestamps of wrong codes]

HEADERS = ["Date", "Time", "Name", "Email", "Verified with", "Video description", "Video file", "Size (MB)",
           "Length (sec)", "Spoken language", "Translated to", "Voice", "Dubbed", "Clip start (s)",
           "Clip length (s)", "Output quality", "Result", "Processing time (sec)", "Note"]

_lock = threading.Lock()
_sessions = {}   # token -> dict
_sends = {}      # email -> [timestamps]
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _load_env():
    path = os.path.join(ROOT, ".env")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8-sig"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()


def mail_ready():
    return bool(os.environ.get("GMAIL_ADDRESS") and os.environ.get("GMAIL_APP_PASSWORD")) \
        or os.environ.get("OTP_DEV_PRINT") == "1"


def _hash(code, salt):
    return hashlib.sha256((salt + code).encode()).hexdigest()


def _mask(email):
    name, _, dom = email.partition("@")
    return (name[:2] + "***@" + dom) if len(name) > 2 else ("***@" + dom)


def _send_mail(to, code, name):
    if os.environ.get("OTP_DEV_PRINT") == "1":  # testing only: print instead of emailing
        print(f"[OTP_DEV_PRINT] code for {to}: {code}", flush=True)
        return
    msg = EmailMessage()
    msg["Subject"] = f"Your Whisper Video Translator code: {code}"
    msg["From"] = os.environ["GMAIL_ADDRESS"]
    msg["To"] = to
    msg.set_content(
        f"Hi {name},\n\nYour verification code is {code}\n"
        f"It works for {OTP_TTL // 60} minutes and for one video translation.\n\n"
        "If you did not ask for this, ignore this email.")
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context(), timeout=20) as s:
        s.login(os.environ["GMAIL_ADDRESS"], os.environ["GMAIL_APP_PASSWORD"].replace(" ", ""))
        s.send_message(msg)


def _show_on_screen(name, email, code):
    """Show the code on THIS computer only: a pop-up window on top of everything (plus the console).
    Whoever sits at the PC reads it and tells/uses it; people on other devices never see it."""
    print(f"[CODE] {name} <{email}> verification code: {code}", flush=True)
    if os.name == "nt" and os.environ.get("OTP_NO_POPUP") != "1":
        import ctypes

        def box():
            ctypes.windll.user32.MessageBoxW(
                0, f"Verification code for {name}\n({email})\n\n          {code}\n\n"
                   f"Valid for {OTP_TTL // 60} minutes, for one video translation.",
                "Whisper Video Translator - code", 0x40 | 0x10000 | 0x40000)  # info icon, foreground, topmost

        threading.Thread(target=box, daemon=True).start()


def send_code(token, name, email, description, channel="email"):
    """channel 'email' (Gmail) or 'screen' (pop-up on this PC). Returns (token, ok, message)."""
    name, email, description = (name or "").strip(), (email or "").strip().lower(), (description or "").strip()
    if len(name) < 2:
        return token, False, "Enter your name."
    if not EMAIL_RE.match(email):
        return token, False, "Enter a valid email address."
    if len(description) < 5:
        return token, False, "Write a short description of the video (at least a few words)."
    if channel == "email" and not mail_ready():
        return token, False, ("Email is not set up yet. Add GMAIL_ADDRESS and GMAIL_APP_PASSWORD "
                              "to the .env file (see SETUP in access_log.py) and restart.")
    now = time.time()
    with _lock:
        s = _sessions.get(token)
        if s and now - s["sent_at"] < RESEND_WAIT:
            return token, False, f"Wait {int(RESEND_WAIT - (now - s['sent_at']))} seconds before asking again."
        recent = [t for t in _sends.get(email, []) if now - t < 3600]
        if len(recent) >= MAX_SENDS_PER_HOUR:
            return token, False, "Too many codes requested for this email. Try again in an hour."
        token = token or secrets.token_urlsafe(16)
        code, salt = f"{secrets.randbelow(10**6):06d}", secrets.token_hex(8)
        _sessions[token] = dict(mode="Email code" if channel == "email" else "Code on PC screen", name=name, email=email, desc=description[:300], salt=salt,
                                hash=_hash(code, salt), expires=now + OTP_TTL, tries=0,
                                sent_at=now, verified=False, verified_at=0)
        _sends[email] = recent + [now]
    if channel == "screen":
        _show_on_screen(name, email, code)
        return token, True, (f"A code is now showing **on the PC screen** (a pop-up window). Read it there "
                             f"and type it below (valid {OTP_TTL // 60} minutes).")
    try:
        _send_mail(email, code, name)
    except Exception as e:
        with _lock:
            _sessions.pop(token, None)
        return token, False, f"Could not send the email: {type(e).__name__}. Check the Gmail address, " \
                             "App Password and internet connection."
    return token, True, f"Code sent to {_mask(email)}. Enter it below (valid {OTP_TTL // 60} minutes)."


# ---------- authenticator app (TOTP, RFC 6238): works offline ----------
def _hotp(secret_b32, counter, digits=6):
    key = base64.b32decode(secret_b32.upper() + "=" * (-len(secret_b32) % 8))
    mac = hmac.new(key, struct.pack(">Q", counter), hashlib.sha1).digest()
    o = mac[-1] & 15
    return f"{(struct.unpack('>I', mac[o:o + 4])[0] & 0x7FFFFFFF) % 10 ** digits:0{digits}d}"


def _totp_match(secret, code, last_step, now=None):
    """Returns the matching 30-second step (accepts +-1 step for clock drift), or None.
    A step that was already used is rejected, so one code cannot be used twice."""
    step = int((now if now is not None else time.time()) // TOTP_STEP)
    for s in (step, step - 1, step + 1):
        if s > last_step and secrets.compare_digest(_hotp(secret, s), code):
            return s
    return None


def _load_users():
    try:
        return json.load(open(USERS, encoding="utf-8"))
    except Exception:
        return {}


def _save_users(users):
    os.makedirs(RECORDS_DIR, exist_ok=True)
    json.dump(users, open(USERS, "w", encoding="utf-8"), indent=2)


def _locked(email):
    now = time.time()
    _fails[email] = [t for t in _fails.get(email, []) if now - t < LOCK_SECS]
    return len(_fails[email]) >= LOCK_AFTER


def _qr_image(uri):
    import segno
    from PIL import Image
    buf = io.BytesIO()
    segno.make(uri, error="m").save(buf, kind="png", scale=6, border=2)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def totp_start(token, name, email, description):
    """Returns (token, ok, message, qr_image_or_None). New email -> QR code to scan once."""
    name, email, description = (name or "").strip(), (email or "").strip().lower(), (description or "").strip()
    if len(name) < 2:
        return token, False, "Enter your name.", None
    if not EMAIL_RE.match(email):
        return token, False, "Enter a valid email address (it is the ID linked to your authenticator app).", None
    if len(description) < 5:
        return token, False, "Write a short description of the video (at least a few words).", None
    with _lock:
        if _locked(email):
            return token, False, "Too many wrong codes for this email. Try again in a few minutes.", None
        token = token or secrets.token_urlsafe(16)
        user = _load_users().get(email)
        base = dict(mode="Authenticator app", name=name, email=email, desc=description[:300],
                    expires=time.time() + 3600, tries=0, sent_at=time.time(), verified=False, verified_at=0)
        if user:
            _sessions[token] = dict(base, enroll=False, secret=user["secret"])
            return token, True, (f"Open your authenticator app, find **{ISSUER} ({email})** and type the "
                                 "current 6-digit code below."), None
        secret = base64.b32encode(secrets.token_bytes(20)).decode().rstrip("=")
        _sessions[token] = dict(base, enroll=True, secret=secret)
    uri = (f"otpauth://totp/{quote(ISSUER)}:{quote(email)}?secret={secret}"
           f"&issuer={quote(ISSUER)}&period={TOTP_STEP}&digits=6")
    return token, True, (f"**First time for {email}.** In Google Authenticator (or Microsoft Authenticator, "
                         f"Authy, ...) tap **+** and scan this QR code. No camera? Enter this key by hand: "
                         f"`{secret}`. Then type the 6-digit code the app shows."), _qr_image(uri)


def _verify_totp(token, s, code):
    email = s["email"]
    if _locked(email):
        _sessions.pop(token, None)
        return False, "Too many wrong codes for this email. Try again in a few minutes."
    users = _load_users()
    last = users.get(email, {}).get("last_step", -1)
    step = _totp_match(s["secret"], code, last)
    if step is None:
        s["tries"] += 1
        _fails.setdefault(email, []).append(time.time())
        if s["tries"] >= MAX_TRIES:
            _sessions.pop(token, None)
            return False, "Too many wrong tries. Start again."
        return False, f"Wrong or already-used code. {MAX_TRIES - s['tries']} tries left."
    if s["enroll"]:
        users[email] = {"secret": s["secret"], "name": s["name"],
                        "created": datetime.now().isoformat(timespec="seconds"), "last_step": step}
        s["enroll"] = False
    else:
        users[email]["last_step"] = step
    _save_users(users)
    s["verified"], s["verified_at"] = True, time.time()
    return True, "Verified. You can translate one video now."


def verify(token, code):
    """Returns (ok, message)."""
    code = re.sub(r"\D", "", code or "")
    with _lock:
        s = _sessions.get(token)
        if not s:
            return False, "Ask for a code first."
        if s.get("mode") == "Authenticator app":
            return _verify_totp(token, s, code)
        if time.time() > s["expires"]:
            _sessions.pop(token, None)
            return False, "The code expired. Ask for a new one."
        if s["tries"] >= MAX_TRIES:
            _sessions.pop(token, None)
            return False, "Too many wrong tries. Ask for a new code."
        if not secrets.compare_digest(_hash(code, s["salt"]), s["hash"]):
            s["tries"] += 1
            return False, f"Wrong code. {MAX_TRIES - s['tries']} tries left."
        s["verified"], s["verified_at"] = True, time.time()
    return True, "Verified. You can translate one video now."


def consume(token):
    """Use up the verification for one translation. Returns the person's record, or None."""
    with _lock:
        s = _sessions.get(token)
        if not s or not s["verified"] or time.time() - s["verified_at"] > VERIFIED_TTL:
            return None
        return _sessions.pop(token)


def video_info(path):
    info = {"file": os.path.basename(path or ""), "size_mb": "", "length": ""}
    try:
        info["size_mb"] = round(os.path.getsize(path) / 1e6, 1)
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip()
        info["length"] = round(float(out), 1)
    except Exception:
        pass
    return info


def write_record(person, vinfo, opts, result, seconds, note=""):
    """Append one row to the Excel file. If Excel has it open, keep a CSV backup instead."""
    now = datetime.now()
    row = [now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), person["name"], person["email"],
           person.get("mode", ""), person["desc"], vinfo["file"], vinfo["size_mb"], vinfo["length"], opts.get("src", ""),
           opts.get("tgt", ""), opts.get("voice", ""), "Yes" if opts.get("dub") else "No",
           opts.get("start", 0), opts.get("len", 0) or "full", opts.get("res", ""), result,
           round(seconds, 1), note]
    os.makedirs(RECORDS_DIR, exist_ok=True)
    with _lock:
        from openpyxl import Workbook, load_workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        for attempt in range(3):
            try:
                if os.path.exists(XLSX):
                    wb = load_workbook(XLSX)
                    ws = wb.active
                else:
                    wb = Workbook()
                    ws = wb.active
                    ws.title = "Translations"
                    ws.append(HEADERS)
                    for c in ws[1]:
                        c.font = Font(bold=True, color="FFFFFF")
                        c.fill = PatternFill("solid", fgColor="5B2A86")
                    ws.freeze_panes = "A2"
                    for col, w in zip("ABCDEFGHIJKLMNOPQRS",
                                      [11, 9, 18, 26, 18, 40, 28, 10, 11, 14, 14, 9, 8, 10, 11, 18, 10, 12, 40]):
                        ws.column_dimensions[col].width = w
                ws.append(row)
                ws.cell(ws.max_row, 6).alignment = Alignment(wrap_text=True, vertical="top")
                wb.save(XLSX)
                return XLSX
            except PermissionError:
                time.sleep(1)
        with open(PENDING_CSV, "a", newline="", encoding="utf-8-sig") as f:  # Excel had the file open
            csv.writer(f).writerow(row)
        return PENDING_CSV
