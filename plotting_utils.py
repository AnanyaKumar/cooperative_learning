
import matplotlib.pyplot as plt
import numpy as np

def visualize_critic(critic):
    pass
    visualize_nn(lambda in_state: critic.predict_on_batch(in_state)[0][0],
        [[(0.5, 0.0), (0.0, 0.0)], [(0.5, 0.5), (0.5, -0.5)]], "critic",
        'gray', None, None, 0.5)

def visualize_actor(actor, max_accel):
    visualize_nn(lambda in_state: actor.predict_on_batch(in_state)[0][1],
        [[(0.5, 0.0), (0.0, 0.0)], [(0.5, 0.5), (0.5, -0.5)]], "actor_x_0.5",
        'seismic', -max_accel, max_accel, 0.5)
    # visualize_nn(lambda in_state: actor.predict_on_batch(in_state)[0][1],
    #     [[(0.5, 0.0), (0.0, 0.0)], [(0.5, 0.5), (0.5, -0.5)]], "actor_x_0.8",
    #     'seismic', -max_accel, max_accel, 0.8)

axarr_dict = {}
def visualize_nn(model, varray, plot_name, cmap, vmin, vmax, obs_y):
    global axarr_dict
    plt.ion()
    xlen = 10
    ylen = 5
    x_space = 3.0 / xlen
    y_space = 1.0 / ylen
    scores = [[0] * xlen for _ in range(ylen)]

    # Get subplots, create them if needed
    if plot_name not in axarr_dict:
        _, axarr = plt.subplots(len(varray), len(varray[0]))
        axarr_dict[plot_name] = axarr
    axarr = axarr_dict[plot_name]

    # Generate subplots
    for i in range(len(varray)):
        for j in range(len(varray[i])):
            (vx, vy) = varray[i][j]
            for x in range(xlen):
                for y in range(ylen):
                    in_state = np.array([[x * x_space, y * y_space, vx, vy]])
                    scores[ylen-y-1][x] = model(in_state)
            axarr[i, j].imshow(scores, cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
            axarr[i, j].set_title(plot_name + ', vx = ' + str(vx) + ', vy = ' + str(vy))

    plt.show()

