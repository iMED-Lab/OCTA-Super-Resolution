"""
Author: Jaborie
Time:   2019.11.03
Intro:  Implement the visdom function in one file.
"""
import visdom
import time
import numpy as np


class Visualizer(object):
    def __init__(self, env='main', port='8097', **kwargs):
        self.vis = visdom.Visdom(env=env, port=port, **kwargs)

        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        """
        @params d: dict (name,value) i.e. ('loss',0.11)
        """
        for k, v in d.items():
            self.plot(k, v)

    def img_many(self, d):
        """
        img several images.

        Args:
            d:
        Returns:
            no returns.
        Raises:
            Not yet encountered
        Examples:
            ...
        """
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        """
        plot the line of given data

        Args:
            name: the window name of plot figure.
        Returns:
            No returns.
        Raises:
            Not yet encountered
        Examples:
            self.plot('loss',1.00)
        """
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1


    def img(self, name, img_, nrow = 8, **kwargs):
        """
        Visualize images.

        Args:
            name: window name
            img:  images needed to show
            nrow: the number of images in one row
        Returns:
            No returns.
        Raises:
            Not yet encounter.
        Examples:
            self.img('input_img',t.Tensor(64,64))
            self.img('input_imgs',t.Tensor(3,64,64))
            self.img('input_imgs',t.Tensor(100,1,64,64))
            self.img('input_imgs',t.Tensor(100,3,64,64),nrows=10)
            ！！！don‘t ~~self.img('input_imgs',t.Tensor(100,64,64),nrows=10)~~！！！
        """
        self.vis.images(img_,
                        nrow=nrow,
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )

    # plot line
    def plot_multi_win(self, d):
        '''
        一次plot多个或者一个
        @params d: dict (name, value) i.e. ('loss', 0.11)
        '''
        for k, v in d.items():
            self.plot(k, v)

    def plot_single_win(self, win, d):
        """
        :param d: dict (name, value) i.e. ('loss', 0.11)
        :param win: only one win
        :param loop_i: i.e. plot testing loss and label
        :return:
        """
        for k, v in d.items():
            index_k = '{}_{}'.format(k, win)
            x = self.index.get(index_k, 0)
            self.vis.line(Y=np.array([v]), X=np.array([x]),
                            name=k,
                            win=win,
                            opts=dict(title=win, showlegend=True),
                            update=None if x == 0 else "append")
            self.index[index_k] = x + 1

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1,'lr':0.0001})
        """

        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        return getattr(self.vis, name)
