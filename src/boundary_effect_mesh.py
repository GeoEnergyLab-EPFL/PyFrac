import json

file = '/home/carlo/BigWhamLink/BigWhamLink/Examples/StaticCrackBenchmarks/boundary_effect_mesh.json'
with open(file) as json_file:
    boundarymesh = json.load(json_file)
    print("json file loaded")