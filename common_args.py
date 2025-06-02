def add_dataset_args(parser):
    parser.add_argument("--n_envs", type=int, required=False,
                        default=100000, help="Envs")
    parser.add_argument("--T", type=int, required=False,
                        default=100, help="Training round")
    parser.add_argument("--envs_eval", type=int, required=False,
                        default=100, help="Eval Envs")
    parser.add_argument("--hists", type=int, required=False,
                        default=1, help="Histories")
    parser.add_argument("--samples", type=int,
                        required=False, default=1, help="Samples")
    parser.add_argument("--horizon", type=int, required=False,
                        default=100, help="horizon")
    parser.add_argument("--dim", type=int, required=False,
                        default=10, help="Dimension")
    parser.add_argument("--context_horizon", type=int, required=False,
                        default=4, help="context horizon")
    parser.add_argument("--rollin_type", type=str, required=False,
                        default='expert', help="behavior mode")
    parser.add_argument("--lin_d", type=int, required=False,
                        default=2, help="Linear feature dimension")
    parser.add_argument("--normal", type=bool, required=False,
                        default=False, help="behavior mode")

    parser.add_argument("--var", type=float, required=False,
                        default=0.0, help="Bandit arm variance")
    parser.add_argument("--cov", type=float, required=False,
                        default=0.0, help="Coverage of optimal arm")

    parser.add_argument("--env", type=str, required=True, help="Environment")
    parser.add_argument("--env_id_start", type=int, required=False,
                        default=-1, help="Start index of envs to sample")
    parser.add_argument("--env_id_end", type=int, required=False,
                        default=-1, help="End index of envs to sample")
    parser.add_argument('--device', type=str, default='cuda:0')

def add_train_args(parser):
    parser.add_argument("--num_epochs", type=int, required=False,
                        default=400000, help="Number of epochs")

def add_model_args(parser):
    parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 123456)')
    parser.add_argument("--beta", type=float, required=False,
                        default=100, help="beta")
    parser.add_argument("--lambda", type=float, required=False,
                        default=0.1, help="beta")
    parser.add_argument("--iql_tau", type=float, required=False,
                    default=0.7, help="iql_tau")
    parser.add_argument("--tau", type=float, required=False,
                    default=0.005, help="tau")
    parser.add_argument("--discount", type=float, required=False,
                    default=0.99, help="discount")
    parser.add_argument("--n_embd", type=int, required=False,
                        default=32, help="Embedding size")
    parser.add_argument("--head", type=int, required=False,
                        default=1, help="Number of heads")
    parser.add_argument("--n_layer", type=int, required=False,
                        default=3, help="Number of layers")
    parser.add_argument("--m_layer", type=int, required=False,
                        default=2, help="Number of layers")
    parser.add_argument("--lr", type=float, required=False,
                        default=1e-3, help="Learning Rate")
    parser.add_argument("--dropout", type=float,
                        required=False, default=0, help="Dropout")
    parser.add_argument('--shuffle', default=False, action='store_true')
    parser.add_argument('--local-rank', default= 0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='batch size (default: 256)')
    parser.add_argument("--decoder_type", type=str, required=False, default='reward')
    parser.add_argument("--c_layer", type=int, required=False,
                        default=1, help="Number of layers")
    parser.add_argument('--context_hidden_dim', type=int, default=128)
    parser.add_argument('--context_dim', type=int, default=16)
    parser.add_argument('--context_batch_size', type=int, default=128)
    parser.add_argument('--context_train_epochs', type=int, default=1000)
    parser.add_argument('--save_context_model_every', type=int, default=200)
    parser.add_argument('--context_lr', type=float, default=0.001)
    parser.add_argument('--context_epoch', type=int, default=200)

def add_eval_args(parser):
    parser.add_argument("--epoch", type=int, required=False,
                        default=-1, help="Epoch to evaluate")
    parser.add_argument("--freq", type=int, required=False,
                        default=100, help="Frequency to evaluate")
    parser.add_argument("--test_cov", type=float,
                        required=False, default=-1.0,
                        help="Test coverage (for bandit)")
    parser.add_argument("--hor", type=int, required=False,
                        default=-1, help="Episode horizon (for mdp)")
    parser.add_argument("--n_eval", type=int, required=False,
                        default=100, help="Number of eval trajectories")
    parser.add_argument("--save_video", default=False, action='store_true')


