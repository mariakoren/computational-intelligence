from simpful import *

FS = FuzzySystem()

TLV = AutoTriangle(3, terms=['poor', 'average', 'good'], universe_of_discourse=[0,10])
FS.add_linguistic_variable("service", TLV)
FS.add_linguistic_variable("quality", TLV)

O1 = TriangleFuzzySet(0,0,13,   term="low")
O2 = TriangleFuzzySet(0,13,25,  term="medium")
O3 = TriangleFuzzySet(13,25,25, term="high")
FS.add_linguistic_variable("tip", LinguisticVariable([O1, O2, O3], universe_of_discourse=[0,25]))

FS.add_rules([
	"IF (quality IS poor) OR (service IS poor) THEN (tip IS low)",
	"IF (service IS average) THEN (tip IS medium)",
	"IF (quality IS good) OR (service IS good) THEN (tip IS high)"
	])

FS.set_variable("quality", 10) 
FS.set_variable("service", 9) 

tip = FS.inference()
# print(tip)

# print(FS._variables['service'])
FS.plot_variable('service', "images/service.png")
FS.plot_variable('quality', "images/quality.png")
FS.plot_variable('tip', "images/tip.png")

FS.produce_figure("images/allvariables.png")


