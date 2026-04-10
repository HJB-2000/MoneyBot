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
