import argparse
import torch

from model.model import ModelBaseStageII, ModelBaseStageIII

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('stage_from', type=str)
    parser.add_argument('stage_to', type=str)

    args = parser.parse_args()

    model_from_class = None

    if args.stage_from == 'II' and args.stage_to == 'III':
        model_from_class = ModelBaseStageII

        model_from = model_from_class.load_from_checkpoint(args.path)

        model_to = ModelBaseStageIII()
        model_to.model.load_state_dict(model_from.state_dict(), strict=False)

    elif args.stage_from == 'III' and args.stage_to == 'II':
        model_from_class = ModelBaseStageII

        model_from = model_from_class.load_from_checkpoint(args.path)

        model_to = ModelBaseStageIII()
        model_to.model.load_state_dict(model_from.state_dict(), strict=False)

    torch.save(model_to, args.output)
