import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def make_curve(x, y, title, x_label, y_label, save_path=None):
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.plot(x, y, color='green')
    # plt.plot(x_axix, train_acys, color='green', label='training accuracy')
    # plt.legend()  # 显示图例

    plt.show()
    if save_path:
        plt.save(save_path)


if __name__=="__main__":
    x = np.arange(0.1, 2*np.pi+0.1, 0.1)
    cos_y = np.cos(x)
    sin_y = np.sin(x)
    tan_y = np.tan(x)
    log_y = np.log(x)
    # make_curve(x, cos_y, 'cos', 'x', 'y')
    # make_curve(x, sin_y, 'sin', 'x', 'y')

    ''' 显示在一张图上
    plt.title('test')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x, cos_y, color='green', label='cos', marker='^')
    plt.plot(x, sin_y, color='red', label='sin', marker='+')
    plt.legend(loc='upper right')
    plt.show()
    '''
    ''' 显示在不同的图上
    plt.figure(1)
    # plt.title('test')
    # plt.xlabel('x')
    # plt.ylabel('y')
    plt.plot(x, cos_y, color='green', label='cos', marker='^')
    plt.legend(loc='upper right')

    plt.figure(2)
    plt.title('test')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x, sin_y, color='red', label='sin', marker='+')
    plt.legend(loc='upper right')

    plt.show()
    '''

    ''' 画在一多张子图上
    plt.figure(figsize=(15, 15))
    f1 = plt.subplot(2, 2, 1)
    plt.title('sin')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x, sin_y, color='g')

    plt.subplot(2, 2, 2)
    plt.title('sin')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x, cos_y, color='g')

    plt.subplot(2, 2, 3)
    plt.title('tan')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x, tan_y, color='g')

    # plt.subplot(2, 2, 4)
    # plt.title('log')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.plot(x, log_y, color='g')


    plt.show()
    '''

