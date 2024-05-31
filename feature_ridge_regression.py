import marimo

__generated_with = "0.6.13"
app = marimo.App(width="full")


@app.cell
def __():
    from regression import FeatureRidgeRegression
    import matplotlib.pyplot as plt
    import marimo as mo
    import keras
    import numpy as np
    from scipy.linalg import sqrtm
    import polars as pl
    from pyobsplot import Plot, js
    from tqdm.keras import TqdmCallback
    return (
        FeatureRidgeRegression,
        Plot,
        TqdmCallback,
        js,
        keras,
        mo,
        np,
        pl,
        plt,
        sqrtm,
    )


@app.cell
def __(FeatureRidgeRegression, Plot, js, keras, np, pl, plt):
    def rational_log_space(max_val, denom=10, num_val=100):
        """
        Create a logarithmic space with rational numbers.
        :param max_val: Maximum value of the space.
        :param denom: Denominator of the rational numbers.
        :param num_val: Number of values in the space.
        :return: The logarithmic space.
        """
        return (np.unique(np.geomspace(1, max_val * denom + 1, num_val, dtype=int)) - 1) / denom


    def rational_pow_space(max_val, denom=10, power=2, num_val=100):
        """
        Create a power space with rational numbers.
        :param max_val: Maximum value of the space.
        :param denom: Denominator of the rational numbers.
        :param power: Power of the space.
        :param num_val: Number of values in the space.
        :return: The power space.
        """
        return np.unique(np.round(np.linspace(0, (max_val * denom) ** (1 / power), num_val) ** power)) / denom


    def normalize(images):
        mz = images - np.mean(images, axis=0)
        return mz / mz.std()


    def random_split(prop, images, labels):
        rng = np.random.default_rng()
        choice = rng.choice(list(prop.keys()), p=list(prop.values()), size=len(labels))
        return {k: (images[choice == k], labels[choice == k]) for k in prop.keys()}


    class SaveFeatures(keras.callbacks.Callback):
        def __init__(self, save_callback):
            super().__init__()
            self.log = []
            self.n_epochs = 0
            self.n_steps = 0
            self.save_callback = save_callback

        def save(self):
            if res := self.save_callback(self.model, n_epoch=self.n_epochs, n_step=self.n_steps):
                self.log.append(res)

        def on_epoch_begin(self, epoch, logs=None):
            self.n_epochs = epoch
            self.n_steps = 0

        def on_train_batch_begin(self, batch, logs=None):
            self.save()
            self.n_steps += 1

        def on_train_end(self, logs=None):
            self.n_epochs += 1
            self.n_steps = 0
            self.save()


    def train_model(model, train_data, test_data, n_epochs=10, batch_size=32, n_callbacks=10, callbacks=None, verbose=0, patience=10):
        num_steps = np.ceil(len(train_data[1]) / batch_size)
        model.compile(optimizer="adam", loss="mean_squared_error")
        callbacks = callbacks or dict()
        callback_steps = rational_log_space(n_epochs, denom=num_steps, num_val=n_callbacks)

        def save_callback(keras_model, n_epoch=0, n_step=0):
            if n_epoch + n_step / num_steps not in callback_steps:
                return
            acc = keras_model.evaluate(*test_data, verbose=verbose)
            return {"loss": acc, "epoch": n_epoch, "step": n_step, "epoch+step": n_epoch + n_step / num_steps, "type": "NN"} | {
                name: fun(keras_model) for name, fun in callbacks.items()
            }

        regression_objs = SaveFeatures(save_callback)
        history = keras.callbacks.History()

        model.fit(
            *train_data,
            epochs=n_epochs,
            validation_data=test_data,
            batch_size=batch_size,
            verbose=verbose,
            shuffle=True,
            callbacks=[regression_objs, history, keras.callbacks.EarlyStopping(patience=patience)],
        )

        return regression_objs.log, history.history


    def regression_from_model(model, train_data, test_data, emp_avg_data, empirical=True):
        fun = FeatureRidgeRegression.empirical if empirical else FeatureRidgeRegression.linearized
        return fun(
            model,
            train_data,
            test_data,
            emp_avg_data,
        )


    def filter_imgs(images, labels, class_names, class_map):
        filtered_idx = [i for i, label in enumerate(labels) if class_names[label] in class_map]
        return images[filtered_idx], np.array([class_map[class_names[label]] for label in labels[filtered_idx]])


    def get_data(
        name="fashion-MNIST",
        class_map=None,
    ):
        if class_map is None:
            class_map = {"Shirt": -1, "T-shirt/top": -1, "Pullover": 1, "Coat": 1}
        if name == "fashion-MNIST":
            fashion_mnist = keras.datasets.fashion_mnist
            (
                (train_images, train_labels),
                (test_images, test_labels),
            ) = fashion_mnist.load_data()
            images = np.concatenate([train_images, test_images])
            labels = np.concatenate([train_labels, test_labels])
            class_names = [
                "T-shirt/top",
                "Trouser",
                "Pullover",
                "Dress",
                "Coat",
                "Sandal",
                "Shirt",
                "Sneaker",
                "Bag",
                "Ankle boot",
            ]
            return filter_imgs(images, labels, class_names, class_map)
        if name == "MNIST":
            mnist = keras.datasets.mnist
            (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
            images = np.concatenate([train_images, test_images])
            labels = np.concatenate([train_labels, test_labels])
            class_names = list(range(10))
            return filter_imgs(images, labels, class_names, class_map)
        raise ValueError("Unsupported Dataset")

    def regressions(
        model,
        train_w_data=None,
        train_data=None,
        emp_avg_data=None,
        test_data=None,
        n_epochs=50,
        batch_size=128,
        n_callbacks=50,
        empirical=True,
        linearized=False,
        patience=10
    ):
        linear_model = keras.Sequential(
            [
                keras.Input(shape=train_data[0].shape[1:]),
                keras.layers.Flatten(),
                keras.layers.Dense(1, use_bias=False),
            ]
        )

        linear_model.compile(
            optimizer="adam",
            loss="mean_squared_error",
        )

        callbacks = {}
        if empirical:
            callbacks["emp"] = lambda model: regression_from_model(model, train_data, test_data, emp_avg_data, empirical=True)
        if linearized:
            callbacks["lin"] = lambda model: regression_from_model(model, train_data, test_data, emp_avg_data, empirical=False)

        NNregressions, history = train_model(
            model,
            train_w_data,
            test_data,
            n_epochs=n_epochs,
            batch_size=batch_size,
            n_callbacks=n_callbacks,
            callbacks=callbacks,
            patience=patience,
        )

        Plot.plot(
            {
                "marks": [
                    Plot.lineY(
                        pl.DataFrame(history).with_columns(pl.Series("epoch", range(len(history["loss"])))).melt(id_vars="epoch"),
                        dict(x="epoch", y="value", stroke=js("d=>d.variable=='loss'?'Train':'Test'")),
                    )
                ],
                "title": "Training History",
                "color": {"legend": True, "type": "categorical"},
                "y": {"type": "log", "label": "Loss"},
                "x": {"label": "Epoch"},
                "grid": True,
            },
            format="html",
        )

        lin_regressions = [{"type": "Linear Regression"} | {key: value(linear_model) for (key, value) in callbacks.items()}]

        return NNregressions + lin_regressions

    def df_from_regressions(regression_objs, dictionary):
        frames = []
        for elmnt in regression_objs:
            frame = None
            for key, val in dictionary.items():
                new_frame = elmnt[key].learningCurve(**val)
                new_frame = new_frame.rename({name: f"{name}_{key}" for name in ["genErrRMT", "genErrEmp"] if name in new_frame.columns})
                frame = frame.join(new_frame, on=("n", "lamb")) if frame is not None else new_frame
            frames.append(frame.with_columns([pl.lit(val).alias(key) for key, val in elmnt.items() if isinstance(val, (float, int, str))]))
        return pl.concat(frames, how="diagonal_relaxed")

    def choose(regs, num):
        arr = [reg for reg in regs if reg["type"] == "NN"]
        return [list(arr)[i] for i in np.round(np.linspace(0, len(arr) - 1, num)).astype(int)] + [
            reg for reg in regs if reg["type"] == "Linear Regression"
        ]
        
    def plot_imgs(imgs, labels=None):
        plt.figure(figsize=(6, 6))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(imgs[i], cmap=plt.cm.binary)
            if labels is not None:
                plt.xlabel(labels[i])
        plt.show()

    format_options = {"font": "SF Pro Display", "margin": "0pt"}
    return (
        SaveFeatures,
        choose,
        df_from_regressions,
        filter_imgs,
        format_options,
        get_data,
        normalize,
        plot_imgs,
        random_split,
        rational_log_space,
        rational_pow_space,
        regression_from_model,
        regressions,
        train_model,
    )


@app.cell
def __(Plot):
    def plot(spec, path):
        format_options = {"font": "SF Pro Display", "margin": "0pt"}
        
        Plot.plot(spec, format="html")
        Plot.plot(
            spec,
            path=f"{path}.pdf",
            format_options=format_options,
        )
        Plot.plot(
            spec,
            path=f"{path}.svg",
            format="svg",
            format_options=format_options,
        )
    return plot,


@app.cell
def __(mo):
    mo.md("# Real Data")
    return


@app.cell
def __(get_data, normalize, plot_imgs, random_split):
    def get_MNIST(class_map):
        f_images, f_labels = get_data(name="MNIST", class_map=class_map)
        f_images = normalize(f_images)
        return random_split({"test": 0.1, "train_reg": 0.25, "train_w": 0.25, "emp_cov": 0.4}, f_images, f_labels)

    real_data = get_MNIST({i: (1 if i % 2 == 0 else -1) for i in range(10)})
    plot_imgs(*real_data["test"])
    return get_MNIST, real_data


@app.cell
def __(keras, real_data, regressions):
    real_regressions = regressions(
        keras.Sequential(
            [
                keras.Input(shape=real_data["test"][0].shape[1:]),
                keras.layers.Flatten(),
                keras.layers.Dense(2000, activation="relu", use_bias=False),
                keras.layers.Dense(2000, activation="relu", use_bias=False),
                keras.layers.Dense(1, use_bias=False),
            ]
        ),
        train_w_data=real_data["train_w"],
        train_data=real_data["train_reg"],
        emp_avg_data=real_data["emp_cov"],
        test_data=real_data["test"],
        n_epochs=100,
        batch_size=128,
        n_callbacks=30,
        empirical=True,
        linearized=False,
    )
    return real_regressions,


@app.cell
def __(mo):
    mo.md("## Fixed $\lambda$")
    return


@app.cell
def __(choose, df_from_regressions, mo, np, real_data, real_regressions):
    df_real_emp = df_from_regressions(
        mo.status.progress_bar(choose(real_regressions, 5)),
        {
            "emp": dict(
                lambdas=[0.1, 1, 10, 100],
                ns=np.unique(np.geomspace(10, 10_000, 200, dtype=int)),
                ns_emp=np.unique(np.geomspace(10, len(real_data["train_reg"][1]) / 2, 10, dtype=int)),
                repeats=20,
            )
        },
    )
    return df_real_emp,


@app.cell
def __(Plot, df_real_emp, pl, plot):
    _plot_vars = {"x": "n", "stroke": "lamb", "fx": "epoch+step"}
    _df_NN = df_real_emp.filter(pl.col("type") == "NN").filter(pl.col("epoch") < 2)
    _df_reg = df_real_emp.filter(pl.col("type") == "Linear Regression")
    plot({
        "marks": [
            Plot.line(_df_NN, _plot_vars | {"y": "genErrRMT_emp"}),
            Plot.line(_df_reg, _plot_vars | {"y": "genErrRMT_emp"}),
            Plot.dot(_df_NN, _plot_vars | {"y": "genErrEmp_emp"}),
            Plot.dot(_df_reg, _plot_vars | {"y": "genErrEmp_emp"}),
            Plot.ruleY(_df_NN, {"y": "loss", "fx": "epoch+step", "stroke": "gray"}),
        ],
        "color": {
            "legend": True,
            "scheme": "viridis",
            "label": "λ",
            "type": "log",
        },
        "width": 400,
        "height": 250,
        "x": {"type": "log", "label": "# Samples"},
        "y": {"domain": [0.08, 4], "label": "Generalization error", "type": "log"},
        "grid": True,
        "fx": {"label": "Epoch"},
    }, "figures/real_emp_small")
    return


@app.cell
def __(Plot, df_real_emp, pl, plot):
    _plot_vars = {"x": "n", "stroke": "lamb", "fx": "epoch+step"}
    _df_NN = df_real_emp.filter(pl.col("type") == "NN")
    _df_reg = df_real_emp.filter(pl.col("type") == "Linear Regression")
    plot({
        "marks": [
            Plot.line(_df_NN, _plot_vars | {"y": "genErrRMT_emp"}),
            Plot.line(_df_reg, _plot_vars | {"y": "genErrRMT_emp"}),
            Plot.dot(_df_NN, _plot_vars | {"y": "genErrEmp_emp"}),
            Plot.dot(_df_reg, _plot_vars | {"y": "genErrEmp_emp"}),
            Plot.ruleY(_df_NN, {"y": "loss", "fx": "epoch+step", "stroke": "gray"}),
        ],
        "color": {
            "legend": True,
            "scheme": "viridis",
            "label": "λ",
            "type": "log",
        },
        "width": 800,
        "height": 300,
        "x": {"type": "log", "label": "# Samples"},
        "y": {"domain": [0.04, 3], "label": "Generalization error", "type": "log"},
        "grid": True,
        "fx": {"label": "Epoch"},
    },"figures/real_emp")
    return


@app.cell
def __(df_from_regressions, mo, np, real_regressions):
    df_real_det = df_from_regressions(
        mo.status.progress_bar(real_regressions),
        {
            "emp": dict(
                lambdas=[0.1, 1, 10],
                ns=np.geomspace(10, 10_000, 200, dtype=int),
            )
        },
    )
    return df_real_det,


@app.cell
def __(Plot, df_real_det, plot):
    plot({
        "marks": [
            Plot.line(
                df_real_det.sort("epoch+step", descending=True),
                dict(x="n", y="genErrRMT_emp", stroke="epoch+step", fx="lamb", opacity=0.8),
            ),
        ],
        "color": {"legend": True, "label": "Epoch", "type": "log"},
        "width": 800,
        "height": 300,
        "x": {"type": "log", "label": "# Samples"},
        "y": {"domain": [0, 1.2], "label": "Generalization error"},
        "grid": True,
        "fx": {"label": "λ"},
    }, "figures/real_det")
    return


@app.cell
def __(mo):
    mo.md("## Optimal $\lambda$")
    return


@app.cell
def __(choose, df_from_regressions, mo, np, real_regressions):
    df_real_opt_emp = df_from_regressions(
        mo.status.progress_bar(choose(real_regressions, 5)),
        {"emp": dict(ns=np.geomspace(30, 5000, 50, dtype=int), ns_emp=np.geomspace(30, 5000, 15, dtype=int), repeats=5)},
    )
    return df_real_opt_emp,


@app.cell
def __(Plot, df_real_opt_emp, js, pl, plot):
    plot({
        "marks": [
            Plot.line(df_real_opt_emp.filter(pl.col("type") == "NN"), dict(x="n", y="genErrRMT_emp", opacity=1, fx="epoch+step")),
            Plot.dot(df_real_opt_emp.filter(pl.col("type") == "NN"), dict(x="n", y="genErrEmp_emp", stroke="lamb", fx="epoch+step")),
            Plot.ruleY(df_real_opt_emp.filter(pl.col("type") == "NN"), dict(y="loss", fx="epoch+step", strokeDasharray=[5, 5])),
            Plot.line(df_real_opt_emp.filter(pl.col("type") == "Linear Regression"), dict(x="n", y="genErrRMT_emp", fx=js("d=>'Linear Regression'"))),
            Plot.dot(df_real_opt_emp.filter(pl.col("type") == "Linear Regression"), dict(x="n", y="genErrEmp_emp", stroke="lamb", fx="type")),
        ],
        "x": {"type": "log", "label": "# Samples"},
        "y": {"domain": [0.04, 0.8], "label": "Generalization error", "type": "log"},
        "grid": True,
        "fx": {"label": "Epoch"},
        "width": 500,
        "height": 400,
        "color": {"legend": True, "type": "log", "label": "λ"},
    },"figures/real_opt_emp")
    return


@app.cell
def __(df_from_regressions, mo, np, real_regressions):
    df_real_opt_det = df_from_regressions(
        mo.status.progress_bar(real_regressions),
        {"emp": dict(ns=np.geomspace(30, 5000, 50, dtype=int))},
    )
    return df_real_opt_det,


@app.cell
def __(Plot, df_real_opt_det, js, pl, plot):
    plot({
        "marks": [
            Plot.line(
                df_real_opt_det.filter(pl.col("type") == "Linear Regression"), dict(x="n", y="genErrRMT_emp", opacity=1, strokeDasharray=[5, 5])
            ),
            Plot.line(
                df_real_opt_det.filter(pl.col("type") == "NN"),
                dict(x="n", y="genErrRMT_emp", stroke="epoch+step", opacity=js("d=>d.epoch+d.step==0?1:.4")),
            ),
        ],
        "x": {"type": "log", "label": "# Samples"},
        "y": {"domain": [0.04, 1.1], "label": "Generalization error", "type": "log"},
        "grid": True,
        "fx": {"label": "λ"},
        "width": 300,
        "height": 400,
        "color": {"legend": True, "type": "log", "label": "Epoch"},
    },"figures/real_opt_det")
    return


@app.cell
def __(mo):
    mo.md("# Synthetic Data")
    return


@app.cell
def __(np, plot_imgs, random_split, real_data):
    def gen_data(data, n_samples):
        sample_dim = data.shape[1:]
        flat_images = data.reshape(data.shape[0], -1)
        pop_cov = flat_images.T @ flat_images / flat_images.shape[0]
        pop_mean = flat_images.mean(axis=0)
        images_artificial = np.random.multivariate_normal(pop_mean, pop_cov, n_samples).reshape(n_samples, *sample_dim)

        w = np.random.normal(0, 1, (images_artificial.shape[1] ** 2, 800)) / 800**0.5
        theta = np.random.normal(0, 1, 800) / 800**0.5
        labels_artificial = np.tanh(images_artificial.reshape(images_artificial.shape[0], -1) @ w) @ theta

        return random_split({"test": 0.1, "train_reg": 0.1, "train_w": 0.4, "emp_cov": 0.4}, images_artificial, labels_artificial)

    synth_data = gen_data(real_data["emp_cov"][0], 100_000)
    plot_imgs(synth_data["test"][0], [f"{x:.2f}" for x in synth_data["test"][1]])
    return gen_data, synth_data


@app.cell
def __(keras, regressions, synth_data):
    synth_regressions = regressions(
        keras.Sequential(
            [
                keras.Input(shape=synth_data["test"][0].shape[1:]),
                keras.layers.Flatten(),
                keras.layers.Dense(1500, activation="tanh", use_bias=False),
                keras.layers.Dense(1400, activation="tanh", use_bias=False),
                keras.layers.Dense(1, use_bias=False),
            ]
        ),
        train_w_data=synth_data["train_w"],
        train_data=synth_data["train_reg"],
        emp_avg_data=synth_data["emp_cov"],
        test_data=synth_data["test"],
        n_epochs=100,
        batch_size=256,
        n_callbacks=50,
        empirical=True,
        linearized=True,
    )
    return synth_regressions,


@app.cell
def __(mo):
    mo.md("## Fixed $\lambda$")
    return


@app.cell
def __(choose, df_from_regressions, mo, np, synth_data, synth_regressions):
    df_art_emp = df_from_regressions(
        mo.status.progress_bar(choose(synth_regressions, 5)),
        {
            "emp": dict(
                lambdas=[0.1, 1, 10, 100],
                ns=np.unique(np.geomspace(50, 10_000, 200, dtype=int)),
                ns_emp=np.unique(np.geomspace(50, len(synth_data["train_reg"][1]), 10, dtype=int)),
                repeats=1,
            ),
            "lin": dict(
                lambdas=[0.1, 1, 10, 100],
                ns=np.unique(np.geomspace(50, 10_000, 200, dtype=int)),
                ns_emp=np.unique(np.geomspace(50, len(synth_data["train_reg"][1]), 10, dtype=int)),
                repeats=1,
            ),
        },
    )
    return df_art_emp,


@app.cell
def __(Plot, df_art_emp, pl, plot):
    _df_NN = df_art_emp.filter(pl.col("type") == "NN")
    _df_reg = df_art_emp.filter(pl.col("type") == "Linear Regression")
    plot({
        "marks": [
            Plot.line(_df_NN, {"x": "n", "stroke": "lamb", "y": "genErrRMT_emp", "fx": "epoch+step"}),
            Plot.line(_df_NN, {"x": "n", "stroke": "lamb", "y": "genErrRMT_lin", "fx": "epoch+step", "strokeDasharray": [5, 5]}),
            Plot.ruleY(_df_NN, {"y": "loss", "fx": "epoch+step", "stroke": "gray"}),
            Plot.dot(_df_NN, {"x": "n", "stroke": "lamb", "y": "genErrEmp_emp", "fx": "epoch+step"}),
            Plot.line(_df_reg, {"x": "n", "stroke": "lamb", "y": "genErrRMT_emp", "fx": "type"}),
            Plot.dot(_df_reg, {"x": "n", "stroke": "lamb", "y": "genErrEmp_emp", "fx": "type"}),
        ],
        "color": {"legend": True, "type": "log", "scheme": "viridis", "label": "λ"},
        "width": 800,
        "height": 400,
        "x": {"type": "log", "label": "# Samples"},
        "y": {"domain": [0.02, 1], "label": "Generalization error", "type": "log"},
        "grid": True,
        "fx": {"label": "Epoch"},
    }, "figures/art_emp")
    return


@app.cell
def __(df_from_regressions, mo, np, synth_regressions):
    df_art_det = df_from_regressions(
        mo.status.progress_bar(synth_regressions),
        {
            "emp": dict(
                lambdas=[0.1, 1, 10],
                ns=np.geomspace(10, 10000, 200, dtype=int),
            )
        },
    )
    return df_art_det,


@app.cell
def __(Plot, df_art_det, plot):
    plot({
        "marks": [
            Plot.line(
                df_art_det.sort("epoch+step", descending=True),
                dict(x="n", y="genErrRMT_emp", stroke="epoch+step", fx="lamb", opacity=0.8),
            ),
        ],
        "color": {"legend": True, "label": "Epoch", "type": "log"},
        "width": 800,
        "height": 500,
        "x": {"type": "log", "label": "# Samples"},
        "y": {"domain": [0.02, 0.5], "label": "Generalization error", "type": "log"},
        "grid": True,
        "fx": {"label": "λ"},
    }, "figures/art_det")
    return


@app.cell
def __(mo):
    mo.md("## Optimal $\lambda$")
    return


@app.cell
def __(choose, df_from_regressions, mo, np, synth_regressions):
    df_art_opt_emp = df_from_regressions(
        mo.status.progress_bar(choose(synth_regressions, 5)),
        {"emp": dict(ns=np.geomspace(30, 5000, 50, dtype=int), ns_emp=np.geomspace(30, 5000, 15, dtype=int), repeats=5)},
    )
    return df_art_opt_emp,


@app.cell
def __(Plot, df_art_opt_emp, js, pl, plot):
    plot({
        "marks": [
            Plot.line(df_art_opt_emp.filter(pl.col("type") == "NN"), dict(x="n", y="genErrRMT_emp", opacity=1, fx="epoch+step")),
            Plot.dot(df_art_opt_emp.filter(pl.col("type") == "NN"), dict(x="n", y="genErrEmp_emp", stroke="lamb", fx="epoch+step")),
            Plot.ruleY(df_art_opt_emp.filter(pl.col("type") == "NN"), dict(y="loss", fx="epoch+step", strokeDasharray=[5, 5])),
            Plot.line(df_art_opt_emp.filter(pl.col("type") == "Linear Regression"), dict(x="n", y="genErrRMT_emp", fx=js("d=>'Linear Regression'"))),
            Plot.dot(df_art_opt_emp.filter(pl.col("type") == "Linear Regression"), dict(x="n", y="genErrEmp_emp", stroke="lamb", fx="type")),
        ],
        "x": {"type": "log", "label": "# Samples"},
        "y": {"domain": [0.02, 0.3], "label": "Generalization error", "type": "log"},
        "grid": True,
        "fx": {"label": "Epoch"},
        "width": 500,
        "height": 400,
        "color": {"legend": True, "type": "log", "label": "λ"},
    }, "figures/art_opt_emp")
    return


@app.cell
def __(df_from_regressions, mo, np, synth_regressions):
    df_art_opt_det = df_from_regressions(
        mo.status.progress_bar(synth_regressions),
        {"emp": dict(ns=np.geomspace(30, 5000, 50, dtype=int))},
    )
    return df_art_opt_det,


@app.cell
def __(Plot, df_art_opt_det, js, pl, plot):
    plot({
        "marks": [
            Plot.line(
                df_art_opt_det.filter(pl.col("type") == "Linear Regression"), dict(x="n", y="genErrRMT_emp", opacity=1, strokeDasharray=[5, 5])
            ),
            Plot.line(
                df_art_opt_det.filter(pl.col("type") == "NN"),
                dict(x="n", y="genErrRMT_emp", stroke="epoch+step", opacity=js("d=>d.epoch+d.step==0?1:.4")),
            ),
        ],
        "x": {"type": "log", "label": "# Samples"},
        "y": {"domain": [0.02, 0.3], "label": "Generalization error", "type": "log"},
        "grid": True,
        "fx": {"label": "λ"},
        "width": 300,
        "height": 400,
        "color": {"legend": True, "type": "log", "label": "Epoch"},
    }, "figures/art_opt_det")
    return


@app.cell
def __(mo):
    mo.md("# Random Features")
    return


@app.cell
def __(FeatureRidgeRegression, np, sqrtm):
    def rf_reg(gamma, d=1000, n_emp=10000, n_test=2000, n_train=2000):
        Ws = np.random.normal(0, d ** (-0.5), (d, d)) @ np.diag(np.linspace(1, d, d) ** (-0.3 / 2))
        W1 = 0.5 * np.random.normal(0, d ** (-0.5), (d, d)) @ np.diag(np.linspace(1, d, d) ** (-gamma / 2)) + 0.5 * Ws
        C3 = np.linalg.inv(W1 @ W1.T + 0.5 * np.eye(d))
        W3 = np.random.normal(0, d ** (-0.5), (d, d)) @ sqrtm(C3)
        theta = np.random.normal(0, d ** (-0.5), d)

        def phis(x):
            return np.array([theta]) @ np.tanh(Ws @ x)

        def phi(x):
            return np.tanh(W3 @ np.tanh(W1 @ np.tanh(W1 @ x)))

        def genData(k):
            x = np.random.normal(0, 1, (d, k))
            return (phi(x), phis(x)[0])

        (Phi_emp, phis_emp) = genData(n_emp)
        Omega = Phi_emp @ Phi_emp.T / n_emp
        sigma = np.mean(phis_emp**2) ** 0.5
        psi = np.mean(phis_emp[np.newaxis] * Phi_emp, axis=1)

        trainData = genData(n_train)
        testData = genData(n_test)

        return FeatureRidgeRegression(Omega=Omega, sigma=sigma, psi=psi, data=trainData, test_data=testData)

    rf_regs = [{"gamma": gamma, "reg": rf_reg(gamma, d=500, n_emp=10_000)} for gamma in [0.01, 0.2, 0.5, 0.8]]
    return rf_reg, rf_regs


@app.cell
def __(Plot, np, pl, plot, rf_regs):
    _df_rf = pl.concat(
        [
            entry["reg"]
            .learningCurve(lambdas=[1e-3], ns=np.linspace(50, 1000, 50, dtype=int), ns_emp=np.linspace(50, 1000, 10, dtype=int), repeats=10)
            .with_columns(pl.lit(entry["gamma"]).alias("gamma"))
            for entry in rf_regs
        ]
    )
    plot({
        "marks": [
            Plot.line(_df_rf, dict(x="n", y="genErrRMT", stroke="gamma", opacity=1)),
            Plot.dot(_df_rf, dict(x="n", y="genErrEmp", stroke="gamma")),
        ],
        "x": {"label": "# Samples"},
        "y": {"type": "log", "label": "Generalization error"},
        "grid": True,
        "width": 400,
        "height": 250,
        "color": {"legend": True, "label": "γ", "scheme": "viridis", "domain": [-0.1, 0.9]},
    }, "figures/rf_emp")
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
