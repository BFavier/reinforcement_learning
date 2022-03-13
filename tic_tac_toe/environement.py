import torch


class Environment:

    def __repr__(self):
        icons = [" X ", "   ", " O "]
        return ("\n"+"-"*11+"\n").join("|".join(icons[i+1] for i in row) for row in self.state)

    def __init__(self):
        self.state = torch.full((3, 3), 0, dtype=torch.long)


if __name__ == "__main__":
    import IPython
    IPython.embed()
