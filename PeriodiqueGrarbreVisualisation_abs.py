import sys

import dash
import dash_cytoscape as cyto
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
from dash.dependencies import Output, Input, State
import re

app = dash.Dash(__name__)
def other(tab,i,elements,k=0,u=0, nLeaf=0, selected=-1):
    tmp = []
    rx = re.search('\((.*)\)\[(.*)]',tab)
    r = rx[1].split(sep=' ')
    elements = elements + [{
                'data': {'id': str(i), 'label': rx[2]},
                'position': {'x': 100 * i, 'y': 50 * k},
            },]
    j = i
    i = i + 1
    k = i
    rm = 0
    for elem in range(0, len(r)):
        if rm > 0:
            rm = rm + len(re.findall('\(', r[elem]))
            rm = rm - len(re.findall('\)', r[elem]))
            tmp[len(tmp)-1] = tmp[len(tmp)-1] + ' ' + r[elem]
        elif r[elem][0] == 'p':
            tmp[len(tmp)-1] = tmp[len(tmp)-1] + ' ' + r[elem]
        elif r[elem][0] == '(':
            rm = len(re.findall('\(', r[elem]))
            rm = rm - len(re.findall('\)', r[elem]))
            tmp.append(r[elem])


        else:
            tmp.append(r[elem])

    for e in range(0, len(tmp)):
        if not tmp[e][0] == '(' and not tmp[e][0] == '[':
            y = i -1
            if selected == nLeaf:
                elements = elements + [{
                    'data': {'id': str(i), 'label': tmp[e], 'leaf': nLeaf},
                    'position': {'x': 100 * y, 'y': 50 * k},
                    'style': {
                        'shape': 'square',
                        'color': px.colors.qualitative.Alphabet[nLeaf],
                        'background-color': px.colors.qualitative.Alphabet[nLeaf],
                    }
                },
                    {
                        'data': {'source': str(i), 'target': str(j)}
                    }
                ]
            else:
                elements = elements + [{
                    'data': {'id': str(i), 'label': tmp[e], 'leaf': nLeaf},
                    'position': {'x': 100 * y, 'y': 50 * k},
                    'style': {
                        'shape': 'square',
                        'background-color': px.colors.qualitative.Alphabet[nLeaf]
                    }
                },
                    {
                        'data': {'source': str(i), 'target': str(j)}
                    }
                ]
            nLeaf = nLeaf+1
            i = i + 1

        if tmp[e][0] == '(':
            elements = elements + [{
                    'data': {'source': str(i), 'target': str(j)}
                }
                ]
            t = other(tmp[e], i, elements, k, nLeaf=nLeaf, selected=selected)
            u = i
            elements = t[0]
            i = t[1]
            nLeaf = t[2]

        if e < len(tmp)-1 and tmp[e+1][1] == 'd':
            if tmp[e][0] == '(':
                elements = elements + [{
                    'data': {'source': str(u), 'target': str(i), 'label': tmp[e + 1]},
                    'classes': 'labelled'
                }]
            else:
                elements = elements + [{
                        'data': {'source': str(i-1), 'target': str(i), 'label': tmp[e+1]},
                        'classes': 'labelled'
                    }]
    t = [elements]
    t.append(i)
    t.append(nLeaf)
    return t

# We create a dataFrame with id, label, period, rep, nested levels, deltas we don't care about shifts yet
def createDataFrame(tab, i, row, k=1, u=0):
    tmp = []
    rx = re.search('\((.*)\)\[r=(.*?) p=(.*)]',tab)
    r = rx[1].split(sep=' ')
    ## Create a row with p and r and nN = 1 and d = 0
    row[0].append('')
    row[1].append(rx[3])
    row[2].append(rx[2])
    row[3].append(k)
    row[4].append(0)
    #Do not change this for : we will use this format after for our things
    rm = 0
    for elem in range(0, len(r)):
        if rm > 0:
            rm = rm + len(re.findall('\(', r[elem]))
            rm = rm - len(re.findall('\)', r[elem]))
            tmp[len(tmp)-1] = tmp[len(tmp)-1] + ' ' + r[elem]
        elif r[elem][0] == 'p':
            tmp[len(tmp)-1] = tmp[len(tmp)-1] + ' ' + r[elem]
        elif r[elem][0] == '(':
            rm = len(re.findall('\(', r[elem]))
            rm = rm - len(re.findall('\)', r[elem]))
            tmp.append(r[elem])


        else:
            tmp.append(r[elem])
    #In this for :
                #   add the label at the right place (last row)
                #   add a row for each d =
    for e in range(0, len(tmp)):
        if not tmp[e][0] == '(' and not tmp[e][0] == '[':
            row[0][len(row[0])-1] = tmp[e]

        if tmp[e][0] == '(':
            k = k+1
            row = createDataFrame(tmp[e],i,row,k,u)
            k = k-1

        if tmp[e][1] == 'd':
            te = re.search('d=(.*)',tmp[e])
            row[0].append('')
            row[1].append(0)
            row[2].append(0)
            row[3].append(k)
            row[4].append(te[1])

    return row

def computeShiftsNested(row, tabE, tabt, tabY):
    t0 = re.search('=(\d+)', tabt)
    t0 = int(t0[1])
    lE = re.split(' ', tabE)
    # Tabs to create DataFrame we insert first elem later
    x = []
    y = []
    label = []
    ids = []
    size = []
    shift = []
    rep = []
    timeX = re.split('; ', tabY)
    timeX[0] = timeX[0][3:]
    timeS = re.split('; ', tabE)
    timeS[0] = timeS[0][7:]

    for elem in range(len(row[2])):
        if row[3][elem] == 1:
            for nRep in range(int(row[2][elem])):
                i = elem + 1
                if not row[0][elem] == '':
                    x.append(timeX[len(x)])
                    y.append(1 - row[3][elem] - nRep * 0.05)
                    ids.append(str(elem))
                    label.append(row[0][elem])
                    size.append(12)
                    shift.append(timeS[len(shift)])
                    rep.append(nRep)

                while i < len(row[2]):
                    if row[3][i] > row[3][elem]:
                        nLevels(row, t0, lE, x, y, label, ids, i, timeX[len(x)], timeX, shift, timeS, rep, nRep)
                    elif not row[0][i] == '':
                        x.append(timeX[len(x)])
                        y.append(1 - row[3][i] - nRep * 0.05)
                        ids.append(str(i))
                        label.append(row[0][i])
                        size.append(12)
                        shift.append(timeS[len(shift)])
                        rep.append(nRep)

                    i = i + 1

    while len(size) < len(x):
        size.append(12)
    df = pd.DataFrame({
        "time": x,
        "y": y,
        "id": ids,
        "label": label,
        "size": size,
        "ideal": shift,
        "nRep": rep,
    })
    return df

def nLevels(row, t0, lE, x, y, label, ids, elem, tmpTime, timeX, shift, timeS, rep, occ):

    for nRep in range(int(row[2][elem])):
        i = elem + 1

        if not row[0][elem] == '':
            x.append(timeX[len(x)])
            y.append(1 - row[3][elem])
            ids.append(str(elem))
            label.append(row[0][elem])
            shift.append(timeS[len(shift)])
            rep.append(occ)

        while i < len(row[2]):
            if row[3][i] > row[3][elem]:
                nLevels(row, t0, lE, x, y, label, ids, i, tmpTime, timeX, shift, timeS, rep, occ)

            i = i + 1

@app.callback(
    Output('cytoscape', 'elements'),
    Input('basic-interactions', 'hoverData'))
def update_graph_1(clickData):
    if clickData:
        e = clickData['points'][0]
        id = re.match('(\d)', e['customdata'][0])
        elements = []
        element = other(tab[1], 0, elements, k=0, u=0, nLeaf=0, selected=int(id[1]))
        return element[0]
    else:
        elements = []
        element = other(tab[1], 0, elements, k=0, u=0, nLeaf=0, selected=-1)
        return element[0]


@app.callback(
    Output('basic-interactions', 'figure'),
    Input('cytoscape', 'mouseoverNodeData'))
def update_time_line(data):
    row = createDataFrame(tab[1], i, [[],[],[],[],[]])
    df = computeShiftsNested(row, tab[6], tab[0], tab[7])
    if data:
        idClicked = 0
        compteur = 0
        while compteur <= data['leaf'] and data['leaf']>0:
            if not row[0][idClicked] == '':
                compteur = compteur + 1
            idClicked = idClicked + 1
        if idClicked >0:
            idClicked = idClicked - 1
        if idClicked == 0 and row[0][idClicked] == '':
            idClicked = idClicked + 1
        for rows in range(df.shape[0]):
            if int(df.at[rows, 'id']) == idClicked:
                df.at[rows,'dot'] = 20
            else:
                df.at[rows,'dot'] = 10

        figure = px.scatter(df, x="time", y="y", color="id", color_discrete_sequence=px.colors.qualitative.Alphabet, hover_name="label", size="dot",
                         hover_data={
                             "y": False,
                             "id":False,
                             "ideal":True,
                             "nRep": True,
                             "dot":False
                         })
        pover = 1
        minMax = [df.at[0, "time"]]
        for r in range(1, df.shape[0]):
            if int(df.at[r, "nRep"]) == int(df.at[r-1, "nRep"]):
                minMax.append(df.at[r,"time"])
            if int(df.at[r, "nRep"]) > int(df.at[r - 1, "nRep"]):# or int(df.at[r, "id"]) == 0:
                figure.add_shape(type='line',
                                     x0=min(minMax),
                                     y0=0.07 - pover * 0.05,
                                     x1=max(minMax),
                                     y1=0.07 - pover * 0.05,
                                     line=dict(color="limegreen", ),
                                     xref='x',
                                     yref='y'
                                     )
                figure.add_shape(type='line',
                                 x0=max(minMax),
                                 y0=1 - pover * 0.05,
                                 x1=max(minMax),
                                 y1=-1 - pover * 0.05,
                                 line=dict(color="salmon", dash="dash"),
                                 xref='x',
                                 yref='y'
                                 )
                pover = pover + 1
                minMax = [df.at[r,"time"]]

        figure.add_shape(type='line',
                      x0=min(minMax),
                      y0=0.07 - pover*0.05,
                      x1=max(minMax),
                      y1=0.07 - pover*0.05,
                      line=dict(color="limegreen", ),
                      xref='x',
                      yref='y'
                      )
        figure.add_shape(type='line',
                      x0=max(minMax),
                      y0=1 - pover*0.05,
                      x1=max(minMax),
                      y1=-1 - pover*0.05,
                      line=dict(color="salmon",dash="dash"),
                      xref='x',
                      yref='y'
                      )
        figure.update_layout(clickmode='event+select')
        figure.update_layout(showlegend=False)
        return figure

    else:
        figure = px.scatter(df, x="time", y="y", color="id", color_discrete_sequence=px.colors.qualitative.Alphabet, hover_name="label",
                         hover_data={
                             "y": False,
                             "id":False,
                             "ideal":True
                         },)

        pover = 1

        minMax = [df.at[0, "time"]]
        for r in range(1,df.shape[0]):
            if int(df.at[r, "nRep"]) == int(df.at[r-1, "nRep"]):
                minMax.append(df.at[r,"time"])
            if int(df.at[r, "nRep"]) > int(df.at[r - 1, "nRep"]):  # or int(df.at[r, "id"]) == 0:
                figure.add_shape(type='line',
                                     x0=min(minMax),
                                     y0=0.07 - pover * 0.05,
                                     x1=max(minMax),
                                     y1=0.07 - pover * 0.05,
                                     line=dict(color="limegreen", ),
                                     xref='x',
                                     yref='y'
                                     )
                figure.add_shape(type='line',
                                     x0=max(minMax),
                                     y0=1 - pover * 0.05,
                                     x1=max(minMax),
                                     y1=-1 - pover * 0.05,
                                     line=dict(color="salmon", dash="dash"),
                                     xref='x',
                                     yref='y'
                                     )
                pover = pover + 1
                minMax = [df.at[r, "time"]]

        figure.add_shape(type='line',
                             x0=min(minMax),
                             y0=0.07 - pover * 0.05,
                             x1=max(minMax),
                             y1=0.07 - pover * 0.05,
                             line=dict(color="limegreen", ),
                             xref='x',
                             yref='y'
                             )
        figure.add_shape(type='line',
                             x0=max(minMax),
                             y0=1 - pover * 0.05,
                             x1=max(minMax),
                             y1=-1 - pover * 0.05,
                             line=dict(color="salmon", dash="dash"),
                             xref='x',
                             yref='y'
                             )
        figure.update_layout(clickmode='event+select')
        figure.update_layout(showlegend=False)
        return figure

if __name__ == "__main__":

    # Real Data
    #t = "t0=2012-06-12 20:30:00	(Personal-Food-Dinner_INS)[r=8 p=198D,0:00]	Code length=69.682748	sum(|E|)=8	Occs=8/8	type=simple	tXstar=2012-06-12 20:30:00; 2012-12-27 20:30:00; 2013-07-13 20:30:00; 2014-01-27 20:30:00; 2014-08-13 20:30:00; 2015-02-27 20:30:00; 2015-09-13 20:30:00; 2016-03-29 20:30:00	tX=2012-06-12 20:30:00; 2012-12-27 20:30:00; 2013-07-13 16:30:00; 2014-01-27 21:30:00; 2014-08-13 21:30:00; 2015-02-27 20:30:00; 2015-09-13 20:30:00; 2016-03-29 22:30:00	event=320_I; 320_I; 320_I; 320_I; 320_I; 320_I; 320_I; 320_I	M"
    #t = "t0=2011-11-28 08:30:00	((Personal-Walk_INS)[r=4 p=7D,0:00])[r=4 p=11:00]	Code length=120.301617	sum(|E|)=8	Occs=16/16	type=nested	tXstar=2011-11-28 08:30:00; 2011-12-05 08:30:00; 2011-12-12 08:30:00; 2011-12-19 08:30:00; 2011-11-28 19:30:00; 2011-12-05 19:30:00; 2011-12-12 19:30:00; 2011-12-19 19:30:00; 2011-11-29 06:30:00; 2011-12-06 06:30:00; 2011-12-13 06:30:00; 2011-12-20 06:30:00; 2011-11-29 17:30:00; 2011-12-06 17:30:00; 2011-12-13 17:30:00; 2011-12-20 17:30:00	tX=2011-11-28 08:30:00; 2011-12-05 08:30:00; 2011-12-12 08:30:00; 2011-12-19 08:30:00; 2011-11-28 19:30:00; 2011-12-05 16:30:00; 2011-12-12 19:30:00; 2011-12-19 19:30:00; 2011-11-29 08:30:00; 2011-12-06 06:30:00; 2011-12-13 06:30:00; 2011-12-20 06:30:00; 2011-11-29 15:30:00; 2011-12-06 16:30:00; 2011-12-13 17:30:00; 2011-12-20 17:30:00	event=31_I; 31_I; 31_I; 31_I; 31_I; 31_I; 31_I; 31_I; 31_I; 31_I; 31_I; 31_I; 31_I; 31_I; 31_I; 31_I	W"
    #t ="t0=2011-11-28 08:30:00	((Business-Work_START)[r=4 p=8D,0:00] [d=7D,0:00] Discretionary-Unpaidwork-Subway_INS [d=1D,1:00] Discretionary-Unpaidwork-Subway_INS)[r=3 p=7D,0:00]	Code length=156.668585	sum(|E|)=6	Occs=18/18	type=other	tXstar=2011-11-28 08:30:00; 2011-12-06 08:30:00; 2011-12-14 08:30:00; 2011-12-22 08:30:00; 2011-12-05 08:30:00; 2011-12-06 09:30:00; 2011-12-05 08:30:00; 2011-12-13 08:30:00; 2011-12-21 08:30:00; 2011-12-29 08:30:00; 2011-12-12 08:30:00; 2011-12-13 09:30:00; 2011-12-12 08:30:00; 2011-12-20 08:30:00; 2011-12-28 08:30:00; 2012-01-05 08:30:00; 2011-12-19 08:30:00; 2011-12-20 09:30:00	tX=2011-11-28 08:30:00; 2011-12-06 08:30:00; 2011-12-14 08:30:00; 2011-12-22 08:30:00; 2011-12-05 08:30:00; 2011-12-06 07:30:00; 2011-12-05 08:30:00; 2011-12-13 08:30:00; 2011-12-21 08:30:00; 2011-12-29 07:30:00; 2011-12-12 08:30:00; 2011-12-13 08:30:00; 2011-12-12 08:30:00; 2011-12-20 08:30:00; 2011-12-28 08:30:00; 2012-01-05 09:30:00; 2011-12-19 08:30:00; 2011-12-20 08:30:00	event=23_S; 23_S; 23_S; 23_S; 121_I; 121_I; 23_S; 23_S; 23_S; 23_S; 121_I; 121_I; 23_S; 23_S; 23_S; 23_S; 121_I; 121_I	W"
    t = sys.argv[1]
    tab = t.split(sep="\t")
    rx = ''
    r = ''
    elements = []
    t0 = 0
    x = 0
    p = 0
    p0= 0
    dE =''
    tXstar = ''
    tX = ''
    for elem in tab:
        if elem[0:2] == "t0":
            t0 = elem
            x = [t0[1]]
            continue
        if elem[0] == "(":
            content = elem
            p = re.split(' ', elem)
            p0 = re.search('p=(.*)]', p[len(p) - 1])
        if elem[0:6] == "tXstar":
            tXstar = elem
            dE = re.split(' ', elem)
        if elem[0:3] == "tX=":
            tX = elem

    # For TimeLine Construction


    y = [0]
    df = pd.DataFrame(columns=['label', 'period', 'nbRep', 'nivNes', 'd'])
    row = [[], [], [], [], []]
    custom = [0]
    label = []
    periode = p0[1]

    i = 0
    elements = []
    row = createDataFrame(content, i, [[], [], [], [], []])
    df = computeShiftsNested(row, tXstar, t0, tX)
    pos = []
    minus = []

    styles = {
        'pre': {
            'border': 'thin lightgrey solid',
            'overflowX': 'scroll'
        }
    }

    app.layout = html.Div(
        [html.H1(str(t0)),
        cyto.Cytoscape(
            id='cytoscape',
            style={'width': '100%', 'height': '600px'},
            layout={'name': 'preset'},
            stylesheet=[
                            {'selector': 'node', 'style': {'label': 'data(label)',
                                                           'text-valign': 'top'}},
                            {'selector': 'edge', 'style': {
                                'curve-style': 'bezier'
                                }
                             },
                            {'selector': '.labelled',
                             'style': {
                                'label': 'data(label)',
                                'line-style': 'dotted',
                                'curve-style': 'bezier',
                                'target-arrow-shape': 'triangle',
                                'arrow-scale': 1
                             },
                            }
                    ]
        ),
        html.Div([
        dcc.Graph(
            id='basic-interactions'
        ),]),
        ]
    )
    app.run_server(debug=False)
