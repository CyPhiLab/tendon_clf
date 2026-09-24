import zmq

from spirob_zmq.core import PUB_ADDR, SUB_ADDR


def _bind_addr(addr):
    if addr.startswith('tcp://') and not addr.startswith('tcp://127.0.0.1'):
        return 'tcp://*:' + addr.rsplit(':', 1)[1]
    return addr


def main():
    ctx = zmq.Context.instance()
    frontend = ctx.socket(zmq.XSUB)
    frontend.bind(_bind_addr(PUB_ADDR))
    backend = ctx.socket(zmq.XPUB)
    backend.bind(_bind_addr(SUB_ADDR))
    try:
        zmq.proxy(frontend, backend)
    except (KeyboardInterrupt, zmq.ContextTerminated):
        pass
    finally:
        frontend.close(0)
        backend.close(0)


if __name__ == '__main__':
    main()
