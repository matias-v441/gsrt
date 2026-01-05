from math import sqrt
tau = (1+sqrt(5))/2
print("golden ratio", tau)
print("-----------------")
a = 2/sqrt(tau*sqrt(5))
print("edge",a)
ri = tau**2*a/2/sqrt(3)
ru = a/2*sqrt(tau*sqrt(5))
print("R inscribed", ri)
print("R circumscribed", ru)
print("-----------------")
a_t = 12/(sqrt(3)*(sqrt(5)+3))
print("edge",a_t)
ri_t = tau**2*a_t/2/sqrt(3)
ru_t = a_t/2*sqrt(tau*sqrt(5))
print("R inscribed", ri_t)
print("R circumscribed", ru_t)
vrt_scale = 0.5*a_t
vx = vrt_scale
vy = tau*vrt_scale
print("grut", sqrt(vx**2+vy**2))
vx = sqrt(.4*(5.+sqrt(5.))) *.5
vy = vx*(1+sqrt(5.))*.5
print("not",sqrt(vx**2+vy**2))