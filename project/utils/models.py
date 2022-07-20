from ..models.gnn import GCN, GAT, SGC, DGCN


def load_model(name, args):
    if name not in ['GCN', 'GAT', 'SGC', 'DGCN']:
        raise ValueError('Invalid model name: {}'.format(name))

    if name == 'GCN':
        model = GCN(args.nfeats, args.nhids, args.nout)
    elif name == 'GAT':
        model = GAT(args.nfeats, args.nhids, args.nout, args.nheads)
    elif name == 'SGC':
        model = SGC(args.nfeats, args.nout, args.K)
    elif name == 'DGCN':
        model = DGCN(args.nfeats, args.nhids, args.nout, args.proj_dim, args.skip_connection)
    
    return model