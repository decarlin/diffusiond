__author__ = 'danielcarlin'

import uuid
import argparse
from bottle import route, run, template, default_app, request, post, abort
import ndex.client as nc
import json
from diffusiond.diffusion import Diffuser
import ndex.networkn as networkn
import base64

parser = argparse.ArgumentParser(description='run the diffusion service')

parser.add_argument('--verbose', dest='verbose', action='store_const',
                    const=True, default=False,
                    help='verbose mode')

arg = parser.parse_args()

if arg.verbose:
    print "Starting diffusion service in verbose mode"
else:
    print "Starting diffusion service"

app = default_app()
app.config['verbose'] = arg.verbose

app.config['ndex'] = nc.Ndex()

current_kernel_id=None
username=None
password=None

@route('/hello')
def index(name='User'):
    verbose_mode = app.config.get("verbose")
    if verbose_mode:
        return template('<b>This is the test method saying Hello verbosely</b>!', name=name)
    else:
        return 'Hello!'


@route('/<network_id>/generate_ndex_heat_kernel', method='GET')
def generate_heat_kernel(network_id):
    return json.dumps({'kernel_id':str(network_id)})

@route('/rank_entities', method='POST')
def diffuse_and_rank():
    dict=json.load(request.body)
    get_these=base64.b64encode(json.dumps(dict.get('identifier_set')))
    kernel_id=dict.get('kernel_id')
    time=dict.get('time')

    if dict.get('normalize') is not None:
        normalize=dict.get('normalize')
    else:
        normalize=False

    if not get_these:
        abort(401, "requires identifier_set parameter in POST data")
    if not kernel_id:
        abort(401, "requires kernel_id parameter in POST data")

    G = networkn.NdexGraph(server='http://public.ndexbio.org', uuid=kernel_id)

    options={'normalize':normalize, 'heatvector':get_these, 'time':time}
    diffused=Diffuser(G,options)
    diffused.start()
    sorted_diffused = sorted(diffused.node_dict.items(), key=operator.itemgetter(1), reverse=True)

    try:
        ranked_heat_by_node_name=[(diffused.network.node[i[0]]['name'],i[1]) for i in sorted_diffused]
    except KeyError:
        ranked_heat_by_node_name=sorted_diffused

    dict_out={"ranked_entities":ranked_heat_by_node_name}
    return json.dumps(dict_out)

run(app, host='0.0.0.0', port=5602)