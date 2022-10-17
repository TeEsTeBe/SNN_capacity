import colorsys
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import seaborn as sns

from colour import Color


def adjust_color(color, lightness_factor=1., saturation_factor=1., lightness_value=None, saturation_value=None):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))

    if lightness_value is not None:
        lightness = lightness_value
    else:
        lightness = max(0., min(1., lightness_factor * c[1]))

    if saturation_value is not None:
        saturation = saturation_value
    else:
        saturation = max(0., min(1., saturation_factor * c[1]))

    return colorsys.hls_to_rgb(c[0], lightness, saturation)


def get_color(name, desaturated=False):
    lightness_fac = 0.7
    saturation_fac = 1.5
    base_colors = {
        'capacity': '#493657',
        'degree': '#73AB84',
        'delay': '#337ca0',
        'accent': '#DA4167',
        'XOR': '#F4E04D',  # '#F77F00',
        'XORXOR': '#042A2B',  # '#A30000',
    }
    assert name in base_colors.keys(), f'No color is defined for {name}. Known names: {list(base_colors.keys())}'

    color = base_colors[name]

    if desaturated:
        color = adjust_color(color, saturation_factor=0.8)
    # else:
    #     color = adjust_color(color, lightness_factor=lightness_fac, saturation_factor=saturation_fac)

    return color


def setup_degree_colors():
    # cmap = plt.get_cmap('tab10')
    # cmap = sns.color_palette("husl", 10)
    # colors = [Color(rgb=cmap[x]).hex for x in range(10)]
    # colors = [Color(rgb=cmap(x)[:-1]).hex for x in range(10)]
    # colors = sns.color_palette(as_cmap=True)
    colors = ['#535c79', '#273d8b',
              '#309975', '#553772',
              '#58b368', '#8f3b76',
              '#dad873', '#c7417b',
              '#efeeb4', '#f5487f',
              ]
    colors.append('#000000')

    return colors

degree_colors = setup_degree_colors()


def get_degree_color(degree):
    color_key = ((degree - 1) % len(degree_colors))
    return degree_colors[color_key]


