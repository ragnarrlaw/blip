"""
moodleSession — the 'saved login', in one place.

holds base/service/credentials, mints the API token once (cached), and hands out both an API
`call()` and a logged-in browser `page`. Both extractors share one session, so login config
lives in exactly one object.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

import moodle.moodle_api as api
from config.conf import Config

log = logging.getLogger("blip.moodle.session")


def _browser_login(page, base: str, user: str, password: str) -> None:
    page.goto(f"{base}/login/index.php", wait_until="domcontentloaded", timeout=1200000)
    page.fill("#username", user)  # standard Moodle login IDs
    page.fill("#password", password)

    page.click("#loginbtn", timeout=1200000)
    page.wait_for_load_state("domcontentloaded", timeout=1200000)

    page.click("#loginbtn")
    page.wait_for_load_state("domcontentloaded")
    if page.locator("#loginbtn").count() > 0:
        raise RuntimeError("browser login failed — the login form is still showing")


class MoodleSession:
    def __init__(self, base: str, service: str, user: str, password: str,
                 *, token_fn=api.get_token, call_fn=api.call):
        self.base = base.rstrip("/")
        self._service, self._user, self._password = service, user, password
        self._token = None
        self._token_fn, self._call_fn = token_fn, call_fn  # injectable for tests

    @classmethod
    def from_config(cls, c: Config, **kw):
        return cls(c.vle_base_url, c.vle_service, c.vle_username, c.vle_password, **kw)

    @property
    def token(self) -> str:
        if self._token is None:
            self._token = self._token_fn(self.base, self._user, self._password, self._service)
        return self._token

    def call(self, fn: str, **params):
        return self._call_fn(self.base, self.token, fn, **params)

    def resolve_course_id(self, shortname: str) -> int:
        return api.resolve_course_id(self.call, shortname)

    def resolve_quiz(self, courseid: int, quiz_name: str) -> api.ResolvedQuiz:
        return api.resolve_quiz(self.call, self.base, courseid, quiz_name)

    @contextmanager
    def browser(self, headless: bool = True):
        """yield a logged-in Playwright page. closes the browser on exit."""
        from playwright.sync_api import sync_playwright  # lazy
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless)
            page = browser.new_page()
            page.on("dialog", lambda d: d.accept())
            try:
                _browser_login(page, self.base, self._user, self._password)
                log.debug("browser session established")
                yield page
            finally:
                browser.close()
