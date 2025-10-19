import numpy as np
import plotly.graph_objects as go
from typing import Tuple
# abc
def add_bias(X: np.ndarray) -> np.ndarray:
    return np.c_[np.ones((X.shape[0], 1)), X]

def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))

def make_logistic_case_2d(cases: int = 1000,
                          difficulty: float = 0,  # 0(易)~1(难)
                          class_bias: float = 0.0,
                          rng: np.random.Generator = np.random.default_rng()
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = rng.normal(0, 1.0, size=(cases, 2))
 
    slope_scale = np.interp(1 - difficulty, [0, 1], [0.6, 6.0])
    noise_std   = np.interp(difficulty,     [0, 1], [0.0, 0.8])

    w_dir = rng.normal(size=2); w_dir /= (np.linalg.norm(w_dir)+1e-12)
    w = slope_scale * w_dir
    b = class_bias
    Theta = np.r_[b, w]  # [b, w1, w2]

    logits = add_bias(X) @ Theta + rng.normal(0, noise_std, size=cases)
    prob = sigmoid(logits)
    Y = (rng.random(cases) < prob).astype(int)
    return X, Y, Theta

class LogisticModel:
    def __init__(self, l2: float = 0.0):
        self.theta = None
        self.l2 = l2

    def fit(self, X: np.ndarray, Y: np.ndarray, epochs: int = 2000, lr: float = 5e-3):
        n, d = X.shape
        Xb = add_bias(X)
        self.theta = np.zeros(d+1)
        for _ in range(epochs):
            p = sigmoid(Xb @ self.theta)
            grad = (Xb.T @ (p - Y)) / n + self.l2 * np.r_[0.0, self.theta[1:]]
            self.theta -= lr * grad

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(add_bias(X) @ self.theta)

def plot_2d(X, Y, theta_true, theta_learned):
    # 网格
    x1_min, x1_max = X[:,0].min()-0.5, X[:,0].max()+0.5
    x2_min, x2_max = X[:,1].min()-0.5, X[:,1].max()+0.5
    xs = np.linspace(x1_min, x1_max, 300)
    ys = np.linspace(x2_min, x2_max, 300)
    Xg1, Xg2 = np.meshgrid(xs, ys)
    Xg = np.c_[Xg1.ravel(), Xg2.ravel()]
    Pg_true = sigmoid(add_bias(Xg) @ theta_true).reshape(Xg1.shape)
    Pg_pred = sigmoid(add_bias(Xg) @ theta_learned).reshape(Xg1.shape)

    fig = go.Figure()

    # 散点（按标签分色）
    for cls, name in [(0, "Class 0"), (1, "Class 1")]:
        mask = (Y == cls)
        fig.add_trace(go.Scatter(
            x=X[mask,0], y=X[mask,1], mode="markers", name=name, opacity=0.7
        ))

  
    fig.add_trace(go.Contour(
        x=xs, y=ys, z=Pg_true, showscale=False,
        contours=dict(start=0.5, end=0.5, size=1.0, coloring="lines"),
        line=dict(width=3), name="True boundary (p=0.5)", hoverinfo="skip"
    ))

    # 学到的边界：p=0.5 等高线
    fig.add_trace(go.Contour(
        x=xs, y=ys, z=Pg_pred, showscale=False,
        contours=dict(start=0.5, end=0.5, size=1.0, coloring="lines"),
        line=dict(dash="dash", width=3), name="Learned boundary (p=0.5)", hoverinfo="skip"
    ))

    fig.update_layout(xaxis_title="x1", yaxis_title="x2", title="Logistic Regression Decision Boundary")
    fig.show()

def main():
    X, Y, Theta_true = make_logistic_case_2d(cases=1200, difficulty=0.2, class_bias=0.0)
    model = LogisticModel(l2=0.0)
    model.fit(X, Y, epochs=3000, lr=5e-3)

    print("True θ:", Theta_true)
    print("Learned θ:", model.theta)
    plot_2d(X, Y, Theta_true, model.theta)

if __name__ == "__main__":
    main()
