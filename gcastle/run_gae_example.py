from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import IIDSimulation, DAG
from castle.algorithms import GAE
import time


def main():
    # data simulation, simulate true causal dag and train_data.
    weighted_random_dag = DAG.erdos_renyi(
        n_nodes=3,
        n_edges=2,
        weight_range=(0.5, 2.0),
        seed=1
    )
    dataset = IIDSimulation(
        W=weighted_random_dag,
        n=100,
        method='linear',
        sem_type='gauss'
    )
    true_causal_matrix, X = dataset.B, dataset.X

    # structure learning with GAE (1 epoch) and timing
    gae = GAE(input_dim=10, epochs=1)
    t0 = time.perf_counter()
    gae.learn(X)
    t1 = time.perf_counter()
    print(f"GAE training time (epochs=1): {t1 - t0:.4f} s")

    # plot predict_dag and true_dag
    GraphDAG(gae.causal_matrix, true_causal_matrix, save_name='result')

    # calculate metrics
    mt = MetricsDAG(gae.causal_matrix, true_causal_matrix)
    print(mt.metrics)


if __name__ == "__main__":
    main()


