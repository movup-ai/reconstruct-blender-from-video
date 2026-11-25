"""One model inference lease across app jobs, released before Blender rendering."""
import contextvars
import functools
import os
import time

_active = contextvars.ContextVar('gpu_lease', default=None)


class Lease:
    def __init__(self, path):
        self.path = path
        self.file = None

    def acquire(self):
        if self.file is not None or not self.path: return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        stream = open(self.path, 'a+b')
        stream.seek(0, os.SEEK_END)
        if not stream.tell(): stream.write(b'0'); stream.flush()
        waiting = False
        try:
            while True:
                stream.seek(0)
                try:
                    if os.name == 'nt':
                        import msvcrt
                        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.file = stream
                    if waiting: print('  model inference slot acquired', flush=True)
                    return
                except (BlockingIOError, OSError):
                    if not waiting: print('  waiting for model inference slot (rendering can continue)', flush=True)
                    waiting = True
                    time.sleep(.2)
        except BaseException:
            stream.close()
            raise

    def release(self):
        stream, self.file = self.file, None
        if stream is None: return
        try:
            stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_UN)
        finally: stream.close()


def acquire():
    lease = _active.get()
    if lease: lease.acquire()


def release():
    lease = _active.get()
    if lease: lease.release()


def job(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        lease = Lease(os.environ.get('BA_GPU_LOCK'))
        token = _active.set(lease)
        try: return function(*args, **kwargs)
        finally:
            lease.release()
            _active.reset(token)
    return wrapped
