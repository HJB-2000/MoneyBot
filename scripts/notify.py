"""
MoneyBot — Gmail email notifier.

HOW GMAIL APP PASSWORDS WORK:
  Gmail blocks direct login from scripts for security.
  Solution: create a dedicated "App Password" in Gmail settings.
  An App Password is a 16-character code that only works for scripts.
  It does NOT give access to the full Gmail account.
  Setup: myaccount.google.com → Security → 2-Step Verification →
         App passwords → Create → name it "moneybot" → copy the code

ENVIRONMENT VARIABLES REQUIRED:
  GMAIL_SENDER   = the Gmail address you send FROM (e.g. you@gmail.com)
  GMAIL_PASSWORD = the 16-char App Password (NOT your real password)
  GMAIL_RECEIVER = the address to send TO (can be same as sender)

  Set them in Codespace terminal before running:
    export GMAIL_SENDER="your@gmail.com"
    export GMAIL_PASSWORD="xxxx xxxx xxxx xxxx"
    export GMAIL_RECEIVER="your@gmail.com"

  Or add them to Codespace Secrets at:
    github.com → Settings → Codespaces → Secrets
  That way they auto-load every time without typing.
"""
from __future__ import annotations

import os
import platform
import smtplib
import traceback
from datetime import datetime, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication


class Notifier:
    def __init__(self):
        self._sender   = os.environ.get("GMAIL_SENDER", "")
        self._password = os.environ.get("GMAIL_PASSWORD", "")
        self._receiver = os.environ.get("GMAIL_RECEIVER", "")

        if self._sender and self._password and self._receiver:
            self.enabled = True
        else:
            self.enabled = False
            print(
                "⚠  Email notifications disabled. Set GMAIL_SENDER, "
                "GMAIL_PASSWORD, GMAIL_RECEIVER to enable."
            )

    def send(self, subject: str, body: str) -> bool:
        if not self.enabled:
            print(f"[notify] {subject}\n{body}")
            return False
        try:
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"]    = self._sender
            msg["To"]      = self._receiver
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                smtp.login(self._sender, self._password)
                smtp.sendmail(self._sender, self._receiver, msg.as_string())
            print(f"✅ Email sent: {subject}")
            return True
        except Exception as e:
            print(f"[notify] Failed to send email: {e}")
            return False

    def send_with_attachment(self, subject: str, body: str,
                             filename: str, file_bytes: bytes) -> bool:
        """Send an email with a file attachment."""
        if not self.enabled:
            print(f"[notify] {subject} (attachment: {filename})")
            return False
        try:
            msg = MIMEMultipart()
            msg["Subject"] = subject
            msg["From"]    = self._sender
            msg["To"]      = self._receiver
            msg.attach(MIMEText(body))
            part = MIMEApplication(file_bytes, Name=filename)
            part["Content-Disposition"] = f'attachment; filename="{filename}"'
            msg.attach(part)
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                smtp.login(self._sender, self._password)
                smtp.sendmail(self._sender, self._receiver, msg.as_string())
            print(f"✅ Email with attachment sent: {subject}")
            return True
        except Exception as e:
            print(f"[notify] Failed to send email with attachment: {e}")
            return False

    def rotate_log(self, log_path: str, max_bytes: int = 3 * 1024 * 1024) -> bool:
        """
        Check if log_path exceeds max_bytes (default 3MB).
        If so: email it as attachment, then truncate the file.
        Returns True if rotation happened.
        """
        try:
            if not os.path.exists(log_path):
                return False
            size = os.path.getsize(log_path)
            if size < max_bytes:
                return False

            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"bot_{ts}.log"

            with open(log_path, "rb") as f:
                log_bytes = f.read()

            subject = f"📋 MoneyBot Log Rotated — {filename}"
            body = (
                f"Bot log reached {size / 1024 / 1024:.1f}MB and was rotated.\n\n"
                f"Timestamp : {datetime.now(timezone.utc).isoformat()}\n"
                f"File      : {filename}\n"
                f"Size      : {size / 1024:.0f} KB\n\n"
                f"The log has been cleared on the server.\n"
                f"Download the attached file to keep a local copy."
            )

            sent = self.send_with_attachment(subject, body, filename, log_bytes)

            # Truncate the log regardless of whether email succeeded
            # (to prevent disk from filling up)
            with open(log_path, "w") as f:
                f.write(f"[{datetime.now(timezone.utc).isoformat()}] Log rotated "
                        f"({size / 1024:.0f}KB archived, emailed={sent})\n")

            print(f"[notify] Log rotated: {size/1024:.0f}KB → emailed={sent}")
            return True
        except Exception as e:
            print(f"[notify] Log rotation failed: {e}")
            return False

    def bot_started(self, capital: float, pairs: int) -> bool:
        ts = datetime.now(timezone.utc).isoformat()
        body = (
            f"MoneyBot has started successfully.\n\n"
            f"Timestamp  : {ts}\n"
            f"Capital    : ${capital:.2f}\n"
            f"Pairs      : {pairs} in scan universe\n"
            f"Mode       : Paper Trading\n\n"
            f"Email notifications are working.\n"
            f"You will receive a heartbeat every 10 hours and the log file when it reaches 3MB."
        )
        return self.send("✅ MoneyBot Started — Email Confirmed", body)

    def heartbeat(self, capital: float, trades: int, wins: int,
                  losses: int, uptime_hours: float) -> bool:
        ts = datetime.now(timezone.utc).isoformat()
        win_rate = f"{wins/(wins+losses)*100:.1f}%" if (wins + losses) > 0 else "N/A"
        body = (
            f"MoneyBot is running normally.\n\n"
            f"Timestamp   : {ts}\n"
            f"Uptime      : {uptime_hours:.1f} hours\n\n"
            f"── Performance ──\n"
            f"Capital     : ${capital:.2f}\n"
            f"Trades      : {trades}\n"
            f"Wins        : {wins}\n"
            f"Losses      : {losses}\n"
            f"Win Rate    : {win_rate}\n\n"
            f"Next heartbeat in 10 hours."
        )
        return self.send("💓 MoneyBot Heartbeat — Still Running", body)

    def training_started(self, steps: list) -> bool:
        ts  = datetime.now(timezone.utc).isoformat()
        machine = platform.node()
        steps_text = "\n".join(f"  • {s}" for s in steps)
        body = (
            f"MoneyBot training has started.\n\n"
            f"Timestamp : {ts}\n"
            f"Machine   : {machine}\n\n"
            f"Steps that will run:\n{steps_text}\n\n"
            f"Estimated total time: 45-60 minutes.\n\n"
            f"You will receive another email when complete."
        )
        return self.send("🤖 MoneyBot Training Started", body)

    def training_complete(self, results: dict) -> bool:
        ts       = datetime.now(timezone.utc).isoformat()
        duration = results.get("duration_minutes", "?")
        files    = results.get("files", {})

        files_text = "\n".join(f"  {k}: {v}" for k, v in files.items())

        top_signals = results.get("top_signals", [])
        signals_text = ""
        if top_signals:
            signals_text = "\nTop 3 most predictive signals:\n"
            signals_text += "\n".join(f"  {i+1}. {s}" for i, s in enumerate(top_signals[:3]))

        body = (
            f"MoneyBot training completed successfully.\n\n"
            f"Timestamp        : {ts}\n"
            f"Total duration   : {duration} minutes\n\n"
            f"Files created:\n{files_text}\n"
            f"{signals_text}\n"
            f"Next step: Run  python run.py  to start paper trading."
        )
        return self.send("✅ MoneyBot Training Complete — Ready for Paper Trading", body)

    def training_failed(self, step: str, error: str, traceback_str: str) -> bool:
        ts = datetime.now(timezone.utc).isoformat()
        tb_lines = traceback_str.splitlines()[:50]
        tb_text  = "\n".join(tb_lines)
        body = (
            f"MoneyBot training FAILED.\n\n"
            f"Timestamp  : {ts}\n"
            f"Failed step: {step}\n"
            f"Error      : {error}\n\n"
            f"Traceback (first 50 lines):\n"
            f"{'─' * 40}\n"
            f"{tb_text}\n"
            f"{'─' * 40}\n\n"
            f"To retry from this step:\n"
            f"  python scripts/train_bayesian.py --resume\n"
            f"Or to restart from scratch:\n"
            f"  python scripts/train_bayesian.py --force"
        )
        return self.send("❌ MoneyBot Training FAILED — Action Required", body)

    def step_complete(self, step_name: str, duration_sec: float, details: str) -> bool:
        ts = datetime.now(timezone.utc).isoformat()
        body = (
            f"Step completed: {step_name}\n\n"
            f"Timestamp : {ts}\n"
            f"Duration  : {duration_sec:.1f} seconds\n\n"
            f"{details}"
        )
        return self.send(f"📊 MoneyBot Step Complete: {step_name}", body)


if __name__ == "__main__":
    n = Notifier()
    n.send("Test", "MoneyBot notifier is working correctly.")
    print("If you received an email, setup is complete.")
