"""Content digests for caches (fit_people.cached_fit_track: one numeric person fit per digest of its inputs)."""
import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=lambda x: x.tolist(),
                                     separators=(',', ':')).encode()).hexdigest()
