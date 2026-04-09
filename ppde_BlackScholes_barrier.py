import torch
import numpy as np
import argparse
import tqdm
import os
import math
import matplotlib.pyplot as plt

from lib.bsde import PPDE_BlackScholes as PPDE
from lib.options import UpAndOutCall, DownAndOutCall

def sample_x0(batch_size, dim, device, S0=1.0):
    return torch.full((batch_size, dim), S0, device=device)

def write(msg, logfile, pbar):
    pbar.write(msg)
    with open(logfile, "a") as f:
        f.write(msg + "\n")

def train(T, n_steps, d, mu, sigma, depth, rnn_hidden, ffn_hidden,
          max_updates, batch_size, lag, base_dir, device, method,
          option_type, K, B, S0):

    logfile = os.path.join(base_dir, "log.txt")
    ts = torch.linspace(0, T, n_steps + 1, device=device)

    if option_type == "up_out_call":
        option = UpAndOutCall(K=K, B=B, idx_traded=0)
    elif option_type == "down_out_call":
        option = DownAndOutCall(K=K, B=B, idx_traded=0)
    else:
        raise ValueError("Unsupported option_type")

    ppde = PPDE(d, mu, sigma, depth, rnn_hidden, ffn_hidden).to(device)
    optimizer = torch.optim.RMSprop(ppde.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.2)

    pbar = tqdm.tqdm(total=max_updates)
    losses = []

    for idx in range(max_updates):
        optimizer.zero_grad()
        x0 = sample_x0(batch_size, d, device, S0=S0)

        if method == "bsde":
            loss, _, _ = ppde.fbsdeint(ts=ts, x0=x0, option=option, lag=lag)
        else:
            loss, _, _ = ppde.conditional_expectation(ts=ts, x0=x0, option=option, lag=lag)

        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.detach().cpu().item())

        if (idx + 1) % 10 == 0:
            with torch.no_grad():
                x0_test = sample_x0(5000, d, device, S0=S0)
                _, Y, payoff = ppde.fbsdeint(ts=ts, x0=x0_test, option=option, lag=lag)
                mc_price = torch.exp(-mu * ts[-1]) * payoff.mean()

            pbar.update(10)
            write(
                "loss={:.4f}, Monte Carlo price={:.6f}, predicted={:.6f}".format(
                    loss.item(), mc_price.item(), Y[0, 0, 0].item()
                ),
                logfile,
                pbar,
            )

    result = {"state": ppde.state_dict(), "loss": losses}
    torch.save(result, os.path.join(base_dir, "result.pth.tar"))

    x0_eval = sample_x0(1, d, device, S0=S0)
    with torch.no_grad():
        x, _ = ppde.sdeint(ts=ts, x0=x0_eval)

    fig, ax = plt.subplots()
    ax.plot(ts.cpu().numpy(), x[0, :, 0].cpu().numpy())
    ax.set_ylabel(r"$X(t)$")
    fig.savefig(os.path.join(base_dir, "path_eval.pdf"))

    pred, mc_pred = [], []
    for idx, t in enumerate(ts[::lag]):
        pred.append(ppde.eval(ts=ts, x=x[:, :(idx * lag) + 1, :], lag=lag).detach())
        mc_pred.append(ppde.eval_mc(ts=ts, x=x[:, :(idx * lag) + 1, :], lag=lag, option=option, mc_samples=10000))

    pred = torch.cat(pred, 0).view(-1).cpu().numpy()
    mc_pred = torch.cat(mc_pred, 0).view(-1).cpu().numpy()

    fig, ax = plt.subplots()
    ax.plot(ts[::lag].cpu().numpy(), pred, '--', label="LSTM + BSDE + sign")
    ax.plot(ts[::lag].cpu().numpy(), mc_pred, '-', label="MC")
    ax.set_ylabel(r"$v(t,X_t)$")
    ax.legend()
    fig.savefig(os.path.join(base_dir, "BS_barrier_LSTM_sol.pdf"))

    print("THE END")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='./numerical_results/', type=str)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--d', default=1, type=int)
    parser.add_argument('--max_updates', default=5000, type=int)
    parser.add_argument('--ffn_hidden', default=[20, 20], nargs="+", type=int)
    parser.add_argument('--rnn_hidden', default=20, type=int)
    parser.add_argument('--depth', default=3, type=int)
    parser.add_argument('--T', default=1.0, type=float)
    parser.add_argument('--n_steps', default=100, type=int)
    parser.add_argument('--lag', default=10, type=int)
    parser.add_argument('--mu', default=0.05, type=float)
    parser.add_argument('--sigma', default=0.3, type=float)
    parser.add_argument('--method', default="bsde", type=str, choices=["bsde", "orthogonal"])
    parser.add_argument('--option_type', default="up_out_call", type=str,
                        choices=["up_out_call", "down_out_call"])
    parser.add_argument('--K', default=1.0, type=float)
    parser.add_argument('--B', default=1.2, type=float)
    parser.add_argument('--S0', default=1.0, type=float)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available() and args.use_cuda:
        device = f"cuda:{args.device}"
    else:
        device = "cpu"

    results_path = os.path.join(args.base_dir, "BS_barrier", args.method, args.option_type)
    os.makedirs(results_path, exist_ok=True)

    train(
        T=args.T,
        n_steps=args.n_steps,
        d=args.d,
        mu=args.mu,
        sigma=args.sigma,
        depth=args.depth,
        rnn_hidden=args.rnn_hidden,
        ffn_hidden=args.ffn_hidden,
        max_updates=args.max_updates,
        batch_size=args.batch_size,
        lag=args.lag,
        base_dir=results_path,
        device=device,
        method=args.method,
        option_type=args.option_type,
        K=args.K,
        B=args.B,
        S0=args.S0,
    )
  
