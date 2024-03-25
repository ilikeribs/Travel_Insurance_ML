import seaborn as sns
import matplotlib.pyplot as plt


def plot_confusion(mx):
    ax = sns.heatmap(
        mx,
        annot=True,
        fmt="g",
        linewidths=0.5,
        cbar=False,
    )
    ax.set_title("Confusion Matrix Predicted-True Outcomes")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.show()


def plot_categorical_columns(df, categorical_columns, target_column="TravelInsurance"):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 20))
    axes = axes.flatten()

    for i, column in enumerate(categorical_columns):
        sns.countplot(x=column, data=df, ax=axes[i], hue=df[target_column])
        axes[i].set_title(f"{column}")

    plt.show()


def plot_numerical_columns(df, numerical_columns, target_column="TravelInsurance"):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))

    axes = axes.flatten()

    for i, column in enumerate(df[numerical_columns]):
        ax = axes[i]
        sns.boxplot(
            data=df,
            y=column,
            x=df["TravelInsurance"],
            ax=ax,
        )
        ax.set_title(f" {column}")
        ax.set_ylabel(column)

    plt.tight_layout()
    plt.show()


def plot_f1_macro(df, x, y):
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(data=df, y=y, x=x, color="steelblue")

    ax.set_title("F1 Macros Scores With 5 Fold Cross Validation")
    ax.set_xlabel("F1 Macro Score")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f")

    sns.despine(top=False)

    plt.show()
