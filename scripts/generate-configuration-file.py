import uproot
import simplejson as json

FILE_DIR = '/eos/project/d/dshep/CEVA'
JSON_OUT_DIR = '../data/file-configuration.json' 

# Define quarks mass, in GeV
q_mass = {'b': 4.18, 'q': 0.096, 't': 173.1, 'W': 80.39, 'h': 124.97}

# Define radius of a jet
delta_R = {'b': 0.4, 'q': 0.4, 't': 0.8, 'W': 0.8, 'h': 0.8}

# Treshold traverse momentum for the jets
min_pt = {}
for jtype, mass in q_mass.items():
    pt = 2*mass/delta_R[jtype]
    min_pt[jtype] = pt

configuration = {}

for jet_type in ['bb', 'tt', 'hh', 'WW']:

    TOTAL_EVENTS = TOTAL_JETS = 0

    configuration[jet_type] = {}
    configuration[jet_type]['files'] = {}

    print('Processing %s files...' % jet_type)

    for file_number in range(100):
        try:
            jtype = 'RSGraviton_%s_NARROW' % jet_type
            rfile = uproot.open('%s/%s/%s_%s.root' %
                                (FILE_DIR, jtype, jtype, file_number))
            if rfile.keys():

                events = len(rfile['Delphes']['Tower'].array())
                pts = rfile['Delphes']['GenJet']['GenJet.PT'].array()

                TOTAL_EVENTS = TOTAL_EVENTS + events
                for e in range(events):
                    jets = len([i for i in pts[e] if i > min_pt[jet_type[0]]])
                    TOTAL_JETS = TOTAL_JETS + jets

                configuration[jet_type]['files'][str(file_number)] = events
        except:
            pass

    configuration[jet_type]['events'] = TOTAL_EVENTS
    configuration[jet_type]['jets'] = TOTAL_JETS

with open(JSON_OUT_DIR, 'w') as f:
    json.dump(configuration, f)
