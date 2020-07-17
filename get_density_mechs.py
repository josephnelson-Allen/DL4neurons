from neuron import h
# Print the names of all density mechanisms
mt = h.MechanismType(0)
mname  = h.ref('')
integer = int(mt.count())
print(type(integer))
for i in range(integer):
    mt.select(i)
    mt.selected(mname)
    print(mname[0])

def get_mech_globals(mechname):
    ms = h.MechanismStandard(mechname, -1)
    name = h.ref('')
    mech_globals = []
    integer2 = int(ms.count())
    for j in range(integer2):
        ms.name(name, j)
        mech_globals.append(name[0])
    return mech_globals

print(get_mech_globals('ca_ion'))
