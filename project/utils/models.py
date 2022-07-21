from ..models.gnn import DGCN


def load_model(name, args):
    if name not in ['DGCN']:
        raise ValueError('Invalid model name: {}'.format(name))

    model = DGCN(args.nfeats, args.nhids, args.nout, args.proj_dim, args.skip_connection)
    
    return model
