import pickle, json
import instruction_parsing
batch = 4335716
with open(f"../data/trainset_{batch}.pickle", 'rb') as f:
    trainset = pickle.load(f)

res2idx = {r:i for i,r in enumerate(instruction_parsing.resource_dict.keys())}
args = set()
for cmd in sum(map(lambda x: x["program"], trainset.values()), []):
    if cmd.command == instruction_parsing.Commands.MOVE_CONTENTS:
        continue
    args.add(cmd.arg)
arg2idx = {a:i for i,a in enumerate(sorted(args))}


def serialize_command(cmd):
    r = res2idx[cmd.resource]
    if cmd.command == instruction_parsing.Commands.MOVE_CONTENTS:
        a = res2idx[cmd.arg]
    else:
        a = arg2idx[cmd.arg]
    return (cmd.command.value, a, r)

dataset = []
for assignment, val in trainset.items():
    t = [(ins, [serialize_command(cmd) for cmd in val["program"] if cmd.ts==i+1]) for i,ins in enumerate(val["instructions"])]
    dataset.append(t)

with open("../preprocessed/arg2idx.json", 'w') as f:
    json.dump(arg2idx, f)
with open("../preprocessed/res2idx.json", 'w') as f:
    json.dump(res2idx, f)
with open(f"../preprocessed/trainset_{batch}.json", 'w') as f:
    json.dump(dataset, f)