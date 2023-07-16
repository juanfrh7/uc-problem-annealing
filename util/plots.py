import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import pandas as pd

def plot_schedule(schedule, n_hours, n_generators, save_image=False, image_path=None) -> None:
    """
    Plot the schedule of generators over a given number of hours.
    
    Args:
        schedule (list): List of tuples representing the schedule of generators.
                         Each tuple contains the generator index and the hour.
        n_hours (int): Number of hours.
        n_generators (int): Number of generators.
        save_image (bool): Flag to indicate whether to save the image or not. Default is False.
        image_path (str): Path to save the image. Required if save_image is True.
    """

    # Save image of schedule
    x, y = zip(*schedule)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Assign a unique color to each generator
    colors = sns.color_palette('hls', n_generators)

    # Plot scatter points
    for i in range(n_generators):
        ax.scatter([y[j] for j in range(len(y)) if x[j] == i],
                   [x[j] for j in range(len(x)) if x[j] == i],
                   color=colors[i], label=f'Generator {i}')

    width = 1
    height = 1

    # Plot rectangles
    for a_y, a_x in schedule:
        ax.add_patch(Rectangle(
            xy=(a_x - width / 2, a_y - height / 2), width=width, height=height,
            linewidth=1, color=colors[a_y], fill=True))

    ax.axis('equal')
    ax.set_xticks(range(n_hours))
    ax.set_yticks(range(n_generators))
    ax.set_xlabel("Hours")
    ax.set_ylabel("Generators")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if save_image:
        if image_path is None:
            raise ValueError("image_path must be provided when save_image is True.")
        plt.savefig(image_path)
    else:
        plt.show()

def plot_error(dictionary):
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame.from_dict(dictionary, orient='index', columns=['Relative error'])
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Problem Size'}, inplace=True)

    # Plot the data
    plt.figure(figsize=(8, 6))
    plt.grid()
    sns.lineplot(x='Problem Size', y='Relative error', data=df, marker='o')
    plt.xlabel('Problem Size')
    plt.ylabel('Relative Error')
    plt.title('Relative Error by Problem Size')
    plt.show()

def plot_comparison(qpu, cpu):
    # Convert the dictionaries to DataFrames
    df1 = pd.DataFrame.from_dict(qpu, orient='index', columns=['Time'])
    df1.reset_index(inplace=True)
    df1.rename(columns={'index': 'Problem Size'}, inplace=True)
    df1['Method'] = 'Leap Hybrid CQM Solver'

    df2 = pd.DataFrame.from_dict(cpu, orient='index', columns=['Time'])
    df2.reset_index(inplace=True)
    df2.rename(columns={'index': 'Problem Size'}, inplace=True)
    df2['Method'] = 'CBC MILP Solver'

    # Concatenate the DataFrames
    df = pd.concat([df1, df2])

    # Plot the data
    plt.figure(figsize=(8, 6))
    plt.grid()
    sns.lineplot(x='Problem Size', y='Time', hue='Method', style='Method', data=df)
    plt.xlabel('Problem Size')
    plt.ylabel('Time [s]')
    plt.title('Comparison of Times by Problem Size')
    plt.legend(title='Method')
    plt.show()