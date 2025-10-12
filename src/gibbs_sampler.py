"""Script for sampling posterior draws using a Gibbs sampler."""

import os
import json

import numpy as np

from src.utils import fetch_data, load_config

np.random.seed(42)

df = fetch_data(
    {
        'sales': "/workspace/data/sales.xls",
        'display': "/workspace/data/displ.xls",
        'coupon': "/workspace/data/coupon.xls",
        'price': "/workspace/data/price.xls"
    }
)

y = df['sales'].values
X = df.drop(columns='sales').values

cfg = load_config(path="/workspace/src/config/config.yml")
DRAW_COUNT = (cfg['nos']*cfg['nod'])+cfg['nob']

sigma_sq = 1
gamma = 1

beta_draws = np.zeros((DRAW_COUNT, 3))
sigma_sq_draws = np.zeros((DRAW_COUNT, 1))
gamma_draws = np.zeros((DRAW_COUNT, 1)) 

print('\nStarting Gibbs Sampler...')

for i in range(DRAW_COUNT):
    if i%5000==0: print(i)

    term_1 = gamma**2*X.T@X + 0.5*np.eye(X.shape[1])
    term_2 = gamma*X.T@y - gamma**2*X.T@np.ones((len(y)))
    beta_mean = np.linalg.inv(term_1) @ term_2
    beta_cov = sigma_sq * np.linalg.inv(term_1)

    beta = np.random.multivariate_normal(mean=beta_mean, cov=beta_cov)

    z_b = np.ones((len(y))) + X@beta
    gamma_mean = z_b.T@y / (z_b.T@z_b)
    gamma_var = sigma_sq / (z_b.T@z_b)

    gamma = np.random.normal(loc=gamma_mean, scale=np.sqrt(gamma_var))

    z_b = np.ones((len(y))) + X@beta
    param = (y - gamma*z_b).T@(y - gamma*z_b) + 0.5*beta.T@beta
    sample_chisq = np.random.chisquare(len(y) + len(beta))

    sigma_sq = param / sample_chisq

    gamma_draws[i] = gamma
    beta_draws[i] = beta.T
    sigma_sq_draws[i] = sigma_sq

# Remove burn in draws and apply thinning
gamma_draws = gamma_draws[cfg['nob']:]
gamma_draws = gamma_draws[range(0, cfg['nos'] * cfg['nod'], cfg['nod'])]

beta_draws = beta_draws[cfg['nob']:]
beta_draws = beta_draws[range(0, cfg['nos'] * cfg['nod'], cfg['nod'])]

sigma_sq_draws = sigma_sq_draws[cfg['nob']:]
sigma_sq_draws = sigma_sq_draws[range(0, cfg['nos'] * cfg['nod'], cfg['nod'])]

count = 0
for i in range(len(gamma_draws)):
    if beta_draws[i,2] * gamma_draws[i] < 0:
        count += 1
posterior_probability = count / len(gamma_draws)

results = {
    'gamma': {
        '10p': round(np.percentile(gamma_draws, 10), 4),
        'mean': round(np.mean(gamma_draws), 4),
        '90p': round(np.percentile(gamma_draws, 90), 4)
    },
    'beta_1': {
        '10p': round(np.percentile(beta_draws[:,0], 10), 4),
        'mean': round(np.mean(beta_draws[:,0]), 4),
        '90p': round(np.percentile(beta_draws[:,0], 90), 4)
    },
    'beta_2': {
        '10p': round(np.percentile(beta_draws[:,1], 10), 4),
        'mean': round(np.mean(beta_draws[:,1]), 4),
        '90p': round(np.percentile(beta_draws[:,1], 90), 4)
    },
    'beta_3': {
        '10p': round(np.percentile(beta_draws[:,2], 10), 4),
        'mean': round(np.mean(beta_draws[:,2]), 4),
        '90p': round(np.percentile(beta_draws[:,2], 90), 4)
    },
    'sigma_sq': {
        '10p': round(np.percentile(sigma_sq_draws, 10), 4),
        'mean': round(np.mean(sigma_sq_draws), 4),
        '90p': round(np.percentile(sigma_sq_draws, 90), 4)
    },
    'posterior_probability': posterior_probability
}

output_dir = "./results/"
output_path = os.path.join(output_dir, "posterior_quantities.json")
os.makedirs(output_dir, exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(results, f, indent=4)

print(f"\nResults saved to {output_path}")
