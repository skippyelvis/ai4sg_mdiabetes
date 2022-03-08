class Logger:

    def __init__(self, name):
        self.name = name
        self.n = 0

    @property
    def title(self):
        return f"{self.name.upper()},{self.n}"

    def log(self, *args, topbrk="+"*15, btbrk="+"*15):
        msg = ' '.join([str(a) for a in args])
        finalmsg = f"{self.title}: {msg}"
        if topbrk is not None:
            print(topbrk)
        print(finalmsg)
        if btbrk is not None:
            print(btbrk)
            print("")
        self.n += 1

StorageLogger = Logger("storage").log
MainLogger = Logger("main").log
DQNLogger = Logger("dqn").log
AgentLogger = Logger("agent").log
