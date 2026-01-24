INPUT_VALIDATIONS = {
    'audio_url': {
        'type': str,
        'required': True,
    },
    'diarize': {
        'type': bool,
        'required': False,
        'default': False
    },
    'word_timestamps': {
        'type': bool,
        'required': False,
        'default': False
    },
    'webhook_url': {
        'type': str,
        'required': False,
        'default': None
    },
    'webhook_secret': {
        'type': str,
        'required': False,
        'default': None
    },
    'job_id': {
        'type': str,
        'required': False,
        'default': None
    },
}
