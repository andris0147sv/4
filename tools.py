# -*- coding: utf-8 -*-
'''
Модуль со вспомогательными классами и функциями, не связанные напрямую с
методом FDTD
'''

import pylab
import numpy
from typing import List
import numpy.fft as fft


class Probe:
    '''
    Класс для хранения временного сигнала в датчике.
    '''

    def __init__(self, position: int, maxTime: int):
        '''
        position - положение датчика (номер ячейки).
        maxTime - максимально количество временных шагов для хранения в датчике.
        '''
        self.position = position

        # Временные сигналы для полей E и H
        self.E = numpy.zeros(maxTime)
        self.H = numpy.zeros(maxTime)

        # Номер временного шага для сохранения полей
        self._time = 0

    def addData(self, E: List[float], H: List[float]):
        '''
        Добавить данные по полям E и H в датчик.
        '''
        self.E[self._time] = E[self.position]
        self.H[self._time] = H[self.position]
        self._time += 1


class AnimateFieldDisplay:
    '''
    Класс для отображения анимации распространения ЭМ волны в пространстве
    '''

    def __init__(self,
                 maxXSize: int,
                 minYSize: float, maxYSize: float,
                 yLabel: str, dx: float):
        '''
        maxXSize - размер области моделирования в отсчетах.
        minYSize, maxYSize - интервал отображения графика по оси Y.
        yLabel - метка для оси Y.
        dx - размер ячейки разбиения.
        '''
        self.dx = dx
        self.maxXSize = maxXSize
        self.minYSize = minYSize
        self.maxYSize = maxYSize
        self._xList = None
        self._line = None
        self._xlabel = 'x, м'
        self._ylabel = yLabel
        self._probeStyle = 'xr'
        self._sourceStyle = 'ok'

    def activate(self):
        '''
        Инициализировать окно с анимацией
        '''
        self._xList = numpy.arange(self.maxXSize) * self.dx

        # Включить интерактивный режим для анимации
        pylab.ion()

        # Создание окна для графика
        self._fig, self._ax = pylab.subplots()

        # Установка отображаемых интервалов по осям
        self._ax.set_xlim(0, self.maxXSize * self.dx)
        self._ax.set_ylim(self.minYSize, self.maxYSize)

        # Установка меток по осям
        self._ax.set_xlabel(self._xlabel)
        self._ax.set_ylabel(self._ylabel)

        # Включить сетку на графике
        self._ax.grid()

        # Отобразить поле в начальный момент времени
        self._line, = self._ax.plot(self._xList, numpy.zeros(self.maxXSize))

    def drawProbes(self, probesPos: List[int]):
        '''
        Нарисовать датчики.

        probesPos - список координат датчиков для регистрации временных
            сигналов (в отсчетах).
        '''
        # Отобразить положение датчиков
        self._ax.plot([i * self.dx for i in probesPos], [0] * len(probesPos),
                      self._probeStyle)

    def drawSources(self, sourcesPos: List[int]):
        '''
        Нарисовать источники.

        sourcesPos - список координат источников (в отсчетах).
        '''
        # Отобразить положение источников
        self._ax.plot([i * self.dx for i in sourcesPos], [0] * len(sourcesPos),
                      self._sourceStyle)

    def drawBoundary(self, position: int):
        '''
        Нарисовать границу в области моделирования.

        position - координата X границы (в отсчетах).
        '''
        self._ax.plot([position * self.dx, position * self.dx],
                      [self.minYSize, self.maxYSize],
                      '--k')

    def stop(self):
        '''
        Остановить анимацию
        '''
        pylab.ioff()

    def updateData(self, data: List[float], timeCount: int):
        '''
        Обновить данные с распределением поля в пространстве
        '''
        self._line.set_ydata(data)
        self._ax.set_title(str(timeCount))
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()


def showProbeSignals(probes: List[Probe], minYSize: float, maxYSize: float,
                     dt: float):
    '''
    Показать графики сигналов, зарегистрированых в датчиках.

    probes - список экземпляров класса Probe.
    minYSize, maxYSize - интервал отображения графика по оси Y.
    dt - шаг дискретизации по времени.
    '''
    # Создание окна с графиков
    fig, ax = pylab.subplots()

    # Настройка внешнего вида графиков
    ax.set_xlim(0, len(probes[0].E) * dt)
    ax.set_ylim(minYSize, maxYSize)
    ax.set_xlabel('t, c')
    ax.set_ylabel('Ez, В/м')
    ax.grid()
    t = numpy.arange(0, len(probes[0].E)) * dt

    # Вывод сигналов в окно
    for probe in probes:
        if probe == probes[1]:
            probe.E[200:] = 0
            ax.plot(t, probe.E)
        else:
            ax.plot(t, probe.E)

    # Создание и отображение легенды на графике
    legend = ['Probe x = {}'.format(probe.position) for probe in probes]
    ax.legend(legend)

    # Показать окно с графиками
    pylab.show()


def Spectrum(probes: List[Probe], maxTime: int, dt: float,
             Fmin: float, Fmax: float):
    '''
    Расчет спектров падающего и отраженного полей
    Расчет коэффициента отражения в зависимости от частоты
    Fmin - минимальная частота;
    Fmax - максимальная частота.
    '''

    size = 2 ** 16
    df = 1 / (size * dt)
    f = numpy.arange(-size / 2 * df, size / 2 * df, df)

    # Расчет спектра падающего поля
    fall = numpy.zeros(maxTime)
    fall[:200] = probes[1].E[:200]
    fall_spectrum = numpy.abs(fft.fft(fall, size))
    fall_spectrum = fft.fftshift(fall_spectrum)

    # Расчет спектра отраженного поля
    scattered_spectrum = numpy.abs(fft.fft(probes[0].E, size))
    scattered_spectrum = fft.fftshift(scattered_spectrum)

    # Построение графиков
    fig, ax = pylab.subplots()
    ax.plot(f * 1e-9, fall_spectrum / numpy.max(fall_spectrum))
    ax.plot(f * 1e-9, scattered_spectrum / numpy.max(scattered_spectrum))
    ax.grid()
    ax.set_xlim(0, 15e9 * 1e-9)
    ax.set_xlabel('f, ГГц')
    ax.set_ylabel(r'$\frac{S}{S_max}}$')
    ax.legend(['Спектр пад. сигнала', 'Спектр отр. сигнала'], loc=1)
    pylab.show()

    fig, ax = pylab.subplots()
    ax.plot(f * 1e-9, scattered_spectrum / fall_spectrum)
    ax.grid()
    ax.set_xlim(Fmin * 1e-9, Fmax * 1e-9)
    ax.set_ylim(0, 1)
    ax.set_xlabel('f, ГГц')
    ax.set_ylabel('|Г|')
    pylab.show()        
        
