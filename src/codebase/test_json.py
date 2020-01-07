import json

with open('syn_config.json') as config_file:
    data = json.load(config_file)

for i in range(1,19):
	config_number = i
	section = 'DISEASES_15'

	config = data['config'+str(config_number)]
	num_diseases = int(config[section]['num_diseases'])
	clusters = int(config[section]['clusters'])
	size = int(config[section]['size'])
	p = float(config['GLOBAL']['p'])
	q1 = float(config['GLOBAL']['q1'])
	z = float(config['GLOBAL']['z'])

	print(num_diseases, clusters, size, p, q1, z)