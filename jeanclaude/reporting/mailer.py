"""Mailer — sends HTML reports via Gmail SMTP."""
from __future__ import annotations

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)

_SMTP_HOST = "smtp.gmail.com"
_SMTP_PORT = 587


class Mailer:
    """Sends HTML email via Gmail SMTP with STARTTLS.

    Use a Gmail App Password (not account password):
    Google Account → Security → 2-Step Verification → App Passwords.

    Parameters
    ----------
    smtp_user : str
        Gmail address used to send (e.g. ``"you@gmail.com"``).
    smtp_password : str
        Gmail App Password (16-char, no spaces).
    """

    def __init__(self, smtp_user: str, smtp_password: str) -> None:
        self._user = smtp_user
        self._password = smtp_password

    def send(self, html: str, subject: str, recipients: list[str]) -> None:
        """Send the HTML report to all recipients.

        Parameters
        ----------
        html : str
            Full HTML string (as returned by ReportBuilder.build).
        subject : str
            Email subject line.
        recipients : list[str]
            Destination addresses.
        """
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self._user
        msg["To"] = ", ".join(recipients)
        msg.attach(MIMEText(html, "html", "utf-8"))

        with smtplib.SMTP(_SMTP_HOST, _SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(self._user, self._password)
            server.sendmail(self._user, recipients, msg.as_string())

        logger.info("Report sent to %s", recipients)
