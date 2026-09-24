import logging
from fastapi import Depends, HTTPException
from config.conf import Config, get_config
from moodle.moodle_session import MoodleSession

log = logging.getLogger("blip.moodle.dependency")

def get_moodle_session(config: Config = Depends(get_config)) -> MoodleSession:
    """
    FastAPI dependency that provides an authenticated Moodle session.
    """
    try:
        session = MoodleSession.from_config(config)
        # Eagerly fetch the token to validate credentials immediately
        _ = session.token
        return session
    except RuntimeError as e:
        log.error(f"Moodle Authentication Failed: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail=f"Moodle connection failed. Details: {str(e)}"
        )
    except Exception as e:
        log.error(f"Moodle Connection Error: {str(e)}")
        raise HTTPException(
            status_code=502,
            detail="Moodle server is unreachable or timed out."
        )
