#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'qt')


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


# In[3]:


## TODO:
# Make Mirror() take an equation and calculate gradient itself
# Make prop_to_secondary() not need a2 and b2 parameters?


# In[4]:


class Mirror:
    def __init__(self,mtype,surface_eq,grad_x,grad_y,grad_z):
        self.mtype = str.lower(mtype)
        self.surface = surface_eq
        self.grad_x = grad_x
        self.grad_y = grad_y
        self.grad_z = grad_z
     
    #returns unit normal vector at the location of coords
    # coords = array of form [x,y,z]
    def normal(self,coords):
        n = np.asarray([self.grad_x(coords[0]),self.grad_y(coords[1]),self.grad_z(coords[2])])
        return 1/np.linalg.norm(n)*n
    
    #Reflects the ray at the ray's current location and returns the unit vector in the direction of the
    # outgoing reflected ray
    def reflect(self,ray):
        norm = self.normal(ray.get_location())
        ray_unit = ray.unit()
        
        cos_theta = -np.dot(ray_unit,norm)
        
        outgoing_ray = ray_unit + 2*cos_theta*norm
        outgoing_ray_norm = 1/np.linalg.norm(outgoing_ray)*outgoing_ray
                
        return outgoing_ray_norm


# In[5]:


# Propagate currently only works for on-axis rays!!!
class Ray:
    def __init__(self,x_0,y_0,x_dir,y_dir,z_dir):
        self.x_0 = x_0 # initial x location
        self.y_0 = y_0 # initial y location
        self.z_0 = 10 # initial z location
        self.x = self.x_0 # current x location
        self.y = self.y_0 # current y location
        self.z = self.z_0 # current z location
        self.x_dir = x_dir # vector x component
        self.y_dir = y_dir # vector y component
        self.z_dir = z_dir # vector z componnet
        self.length = 1 # vector length
        self.canProp = True # Bool to control if the ray can propagate through the system
    
    #returns ray unit vector
    def unit(self):
        r = self.get_dir()
        # self.get_dir() shoooouuuld always give a unit vector, but might as well be safe? 
        return 1/(np.linalg.norm(r))*r
    
    #Propagates all the way through the telescope returns the final location of ray
    # primary = system primary mirror
    # secondary = system secondary mirror
    # a2, b2 = a^2, b^2 values from secondary equation (blegh, need to get rid of these) 
    # If no focal is provided, assume propagating to focal plane
    def propagate(self,primary,secondary,a2,b2,focal=-0.2):
        self.propagate_to_primary()
        self.set_dir(primary.reflect(self))
        self.propagate_to_secondary(a2,b2)
        self.set_dir(secondary.reflect(self))
        self.propagate_to_focal(focal)
        
        return self.get_location()
      
    #Propagates the ray to the primary mirror and updates its current location
    def propagate_to_primary(self):
        # If the ray can propagate, calculate the length such that it intersects with the primary mirror
        if(self.canProp):
            # If x & y vector components are both 0 there is a division by 0
            if(self.x_dir == 0 and self.y_dir == 0):
                self.length = self.z_0-primary.surface(self.x_0,self.y_0)

            else:
                # Substitute ray coordinates into primary equation and solve for length
                # (z_0+l*z_dir) = ((x_0+l*x_dir)^2+(y_0+l*y_dir)^2)/12
                a = self.x_dir**2+self.y_dir**2
                b = 2*self.x_dir*self.x_0+2*self.y_dir*self.y_0-12*self.z_dir
                c = self.x_0**2+self.y_0**2-12*self.z_0

                self.length = 1/2*(-b/a+np.sqrt((b/a)**2-4*c/a))

            end_loc = np.asarray([self.x_0,self.y_0,self.z_0]+self.length*self.unit())

            self.set_location(end_loc)

            #If the ray falls off of the primary mirror, set its location to 0,0,10
            # and set its canProp attribute to false so it no longer propagates through the system
            if(np.sqrt(self.x**2 + self.y**2) > 0.5 ):
                self.set_location([0,0,10])
                self.canProp = False
        
        return
    
    #Propagates to the secondary mirror and updates its current location
    def propagate_to_secondary(self,a2,b2):
        #We want to propagate to secondary from current location so set origin to location
        if(self.canProp):
            # Update origin so that ray is propagating from its current location to the secondary
            self.set_origin(self.get_location()) 
            
            # Solve for intersection with secondary
            # ((x0+l*x_dir)^2+(y0+l*y_dir)^2)/a^2 - ((z0+l*z_dir)-1.4)^2/b^2 = -1
            a = b2*(self.x_dir**2+self.y_dir**2)-a2*self.z_dir**2
            b = b2*(2*self.x_0*self.x_dir+2*self.y_0*self.y_dir)-2*a2*self.z_dir*(self.z_0-1.4)
            c = b2*(self.x_0**2+self.y_0**2)-a2*(self.z_0-1.4)**2+a2*b2

            self.length = 1/(2*a)*(-b-np.sqrt(b**2-4*a*c))

            end_loc = np.asarray([self.x_0,self.y_0,self.z_0]+self.length*self.unit())

            self.set_location(end_loc)
            
            # If ray falls outside of the secondary reset its location and 
            # set canProp to false to prevent it from propagating through the system
            #if(np.sqrt(self.x**2+self.y**2) > 0.09):
                #self.set_location([0,0,10])
                #self.canProp = False
            
        
        return
    
    #Propagates the ray to the focal plane and updates current position
    def propagate_to_focal(self,focal=-0.2):
        if(self.canProp):
            self.set_origin(self.get_location())
            
            # Solve for length such that z0+l*z_dir = focal
            self.length = (focal-self.z_0)/self.z_dir

            end_loc = np.asarray([self.x_0,self.y_0,self.z_0]+self.length*self.unit())

            self.set_location(end_loc)
        
        return
    
    #Redefines the origin of the ray (Use when proping from primary -> secondary and secondary -> focus)
    # location = array of form [x,y,z]
    def set_origin(self,location):
        self.x_0 = location[0]
        self.y_0 = location[1]
        self.z_0 = location[2]
        return
    
    #returns the initial location of the ray
    def get_origin(self):
        return np.asarray([self.x_0,self.y_0,self.z_0])
    
    #Sets the rays current location to [x,y,z]
    #location = array of form [x,y,z]
    def set_location(self,location):
        self.x = location[0]
        self.y = location[1]
        self.z = location[2]
        return
    
    #returns the current location of the ray
    def get_location(self):
        return np.asarray([self.x,self.y,self.z])
    
    #direction = array of form [x_dir,y_dir,z_dir] ***Must be unit vector***
    # This is only called from Mirror.reflect(), which returns a unit vector
    def set_dir(self,direction):
        self.x_dir = direction[0]
        self.y_dir = direction[1]
        self.z_dir = direction[2]
        return
     
    #returns array with direction components
    # If the ray is set up properly when initialized, the directions should already form a unit vector
    def get_dir(self):
        return np.asarray([self.x_dir,self.y_dir,self.z_dir])


# In[6]:


### Construct equation of primary ###
# (x^2+y^2)/4p - z = 0
# p = focal length
primary_eq = lambda x,y: 1/12*x**2+1/12*y**2
primary_gradx = lambda x: -1/6*x
primary_grady = lambda y: -1/6*y
primary_gradz = lambda z: 1
primary = Mirror('primary',primary_eq,primary_gradx,primary_grady,primary_gradz)


# In[7]:


### Construct equation of secondary ###
# Send a test ray from the edge of the primary to the secondary and find 
# where it intersects with a ray from the focal plane
test_ray = Ray(-0.5,0,0,0,-1)
test_ray.propagate_to_primary()
test_ray.set_dir(primary.reflect(test_ray))
r_hat = test_ray.unit()

# Slope of line from edge of primary
ratio = r_hat[2]/r_hat[0]

# Edge point calculated from intersection of lines from focus & edge of primary
# ie:   where does  ratio*x + 3 = -cos(f/15)/sin(f/15)x - 0.2
x_edge = -3.2/(ratio+np.cos(1/30)/np.sin(1/30))
z_edge = ratio*(x_edge)+3

# Hyperbola of the form x^2/a^2 + y^2/a^2 - (z-k)^2/b^2 = -1
# c^2 = a^2 + b^2, where c = 1/2(distance between foci) = 1.6
# k = center = 3-1.6 = 1.4
# Solve quadratic for a^2, b^2
c = 1.6
k = 1.4
gam = x_edge**2+(z_edge-k)**2-c**2
eta = -c**2*x_edge**2
a2 = 0.5*(-gam+np.sqrt(gam**2-4*eta))
b2 = c**2-a2

# Debugging
print('Edge location: %f' %(x_edge))
print('a^2: %f' %(a2))
print('b^2: %f' %(b2))

# Solving the hyperbola for z gives
# z = k+sqrt(b^2(x^2+y^2/a^2+1))

secondary_eq = lambda x,y: k+np.sqrt(b2*((x**2+y**2)/a2+1))
secondary_gradx = lambda x: 2*x/a2
secondary_grady = lambda y: 2*y/a2
secondary_gradz = lambda z: -2*(z-1.4)/b2

secondary = Mirror('secondary',secondary_eq,secondary_gradx,secondary_grady,secondary_gradz)


# In[8]:


'''
### Test reflection in 3D, make a cool plot ###
off_axis_as = 5*60 # Off axis angle in arcsec
off_axis_rad = off_axis_as*4.8481368e-6 # off axis angle in radian

# Set up meshgrids for plotting mirror surfaces
x = np.linspace(-0.5,0.5,1000)
y = np.linspace(-0.5,0.5,1000)

x_2 = np.linspace(x_edge,-x_edge,1000)
y_2 = np.linspace(x_edge,-x_edge,1000)

# XY = primary meshgrid X2Y2 = secondary meshgrid
X,Y = np.meshgrid(x,y)
X2,Y2 = np.meshgrid(x_2,y_2)

#Ray(x_0,y_0,x_dir,y_dir,z_dir)
ray1 = Ray(0,0.2,np.sin(off_axis_rad),0,-np.cos(off_axis_rad))
ray2 = Ray(-0.1,-0.1,np.sin(off_axis_rad),0,-np.cos(off_axis_rad))
ray3 = Ray(0.2,0,np.sin(off_axis_rad),0,-np.cos(off_axis_rad))
ray4 = Ray(-0.2,0.2,np.sin(off_axis_rad),0,-np.cos(off_axis_rad))

ray1.propagate_to_primary()
ray3.propagate_to_primary()
ray3.propagate_to_primary()
ray4.propagate_to_primary()
# Reflect ray towards secondary
ray1.set_dir(primary.reflect(ray1))
ray2.set_dir(primary.reflect(ray2))
ray3.set_dir(primary.reflect(ray3))
ray4.set_dir(primary.reflect(ray4))


## Debugging ##
#print('ray 1 origin: ' +str(ray1.get_origin()))
#print('ray 1 location: '+str(ray1.get_location()))
#print('ray 1 length: %f' %(ray1.length))
#print('ray 1 direction after reflection: '+str(ray1.get_dir()))
#print('surface location: %f, %f' %(ray1.x,primary.surface(ray1.x,0)))
###


#Normal to surface on primary for plotting
normal = primary.normal(ray1.get_location())
normx = [ray1.x,ray1.x+2*normal[0]]
normy = [ray1.y,ray1.y+2*normal[1]]
normz = [ray1.z,ray1.z+2*normal[2]]


## Debugging ##
#print(norm_vec)
#print('normal vector components: '+str(normal))
#print('Normal vector start location: %f, %f' %(normx,normz))
#print('Normal vector end location: %f, %f' %(norm2x,norm2z))
#print('New ray unit vector: '+str(ray1.unit()))
###

#Initial ray components for plotting
rayx = np.asarray([[ray1.x_0,ray1.x],[ray2.x_0,ray2.x],[ray3.x_0,ray3.x],[ray4.x_0,ray4.x]])
rayy = np.asarray([[ray1.y_0,ray1.y],[ray2.y_0,ray2.y],[ray3.y_0,ray3.y],[ray4.y_0,ray4.y]])
rayz = np.asarray([[ray1.z_0,ray1.z],[ray2.z_0,ray2.z],[ray3.z_0,ray3.z],[ray4.z_0,ray4.z]])

ray1.propagate_to_secondary(a2,b2)
ray2.propagate_to_secondary(a2,b2)
ray3.propagate_to_secondary(a2,b2)
ray4.propagate_to_secondary(a2,b2)

# Normal to secondary for plotting
#normal = secondary.normal(ray1.get_location())
#norm_secondx = [ray1.x,ray1.x+2*normal[0]]
#norm_secondy = [ray1.y,ray1.y+2*normal[1]]
#norm_secondz = [ray1.z,ray1.z+2*normal[2]]

# Ray components from primary to secondary for plotting
ray2x = np.asarray([[ray1.x_0,ray1.x],[ray2.x_0,ray2.x],[ray3.x_0,ray3.x],[ray4.x_0,ray4.x]])
ray2y = np.asarray([[ray1.y_0,ray1.y],[ray2.y_0,ray2.y],[ray3.y_0,ray3.y],[ray4.y_0,ray4.y]])
ray2z = np.asarray([[ray1.z_0,ray1.z],[ray2.z_0,ray2.z],[ray3.z_0,ray3.z],[ray4.z_0,ray4.z]])

# Reflect ray towards focus
ray1.set_dir(secondary.reflect(ray1))
ray1.propagate_to_focal(focal=-0.3)
ray2.set_dir(secondary.reflect(ray2))
ray2.propagate_to_focal(focal=-0.3)
ray3.set_dir(secondary.reflect(ray3))
ray3.propagate_to_focal(focal=-0.3)
ray4.set_dir(secondary.reflect(ray4))
ray4.propagate_to_focal(focal=-0.3)

# Ray components from secondary to focus for plotting
ray3x = np.asarray([[ray1.x_0,ray1.x],[ray2.x_0,ray2.x],[ray3.x_0,ray3.x],[ray4.x_0,ray4.x]])
ray3y = np.asarray([[ray1.y_0,ray1.y],[ray2.y_0,ray2.y],[ray3.y_0,ray3.y],[ray4.z_0,ray4.y]])
ray3z = np.asarray([[ray1.z_0,ray1.z],[ray2.z_0,ray2.z],[ray3.z_0,ray3.z],[ray4.z_0,ray4.z]])

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111,projection='3d')
#ax.plot_surface(X,Y,primary.surface(X,Y))
#ax.plot_surface(X2,Y2,secondary.surface(X2,Y2))

print(rayx.shape)
print(ray2x[0,:])
print(ray2y[0,:])
print(ray2z[0,:])


ax.plot(rayx[0,:],rayy[0,:],rayz[0,:],'C3')
ax.plot(rayx[1,:],rayy[1,:],rayz[1,:],'C4')
ax.plot(rayx[2,:],rayy[2,:],rayz[3,:],'C5')

ax.plot(ray2x[0,:],ray2y[0,:],ray2z[0,:],'C3')
ax.plot(ray2x[1,:],ray2y[1,:],ray2z[1,:],'C4')
ax.plot(ray2x[2,:],ray2y[2,:],ray2z[3,:],'C5')

ax.plot(ray3x[0,:],ray3y[0,:],ray3z[0,:],'C3')
ax.plot(ray3x[1,:],ray3y[1,:],ray3z[1,:],'C4')
ax.plot(ray3x[2,:],ray3y[2,:],ray3z[3,:],'C5')

ax.scatter(0,0,3,c='k')
ax.scatter(0,0,-0.2,c='k')

ax.axes.set_zlim([-0.23,-0.18])
ax.axes.set_xlim([-0.001,0.001])
ax.axes.set_ylim([-0.001,0.001])
'''


# In[95]:


start = time.time()

# Off axis angle in arcsec, radian
off_axis_as = 5*60
off_axis_rad = off_axis_as*4.8481368e-6
print('Off-axis angle: %f rad' %(off_axis_rad))

## Make an array of ray objects and propagate all of them through to the focus
x_locs = np.linspace(-0.5,0.5,100,dtype='float')
y_locs = np.linspace(-0.5,0.5,100,dtype='float')

rays = []

for x_loc in x_locs:
    for y_loc in y_locs:
        # Form the annulus of incoming rays by skipping the loop any time coords fall outside of the primary
        # or inside of the secondary
        if((x_loc**2+y_loc**2 > 0.5) or (x_loc**2+y_loc**2 < 0.09)):
            continue
        else:
            # Otherwise make a ray at x_loc, y_loc
            rays.append(Ray(x_loc,y_loc,np.sin(off_axis_rad),0,-np.cos(off_axis_rad)))


locations = np.zeros((len(rays),3))

# Propagate each ray through the system and get the final location
for i in range(0,len(rays)):
    ray = rays[i]
    locations[i] = ray.propagate(primary,secondary,a2,b2,focal=-0.19955)
    

#Get rid of any of the rays we put up at [0,0,10] with a logical array
rays_to_cull = np.zeros(len(rays),dtype=bool)
for i in range(0,len(rays_to_cull)):
    if(locations[i][2] > 0):
        rays_to_cull[i] = True
        
locations = locations[rays_to_cull==False]
print(str(len(locations))+' made it to the focal plane') # How many rays made it through the system

end = time.time()

print('elapsed time: %f' %(end-start))


# In[47]:


plt.figure(figsize=(8,8))
plt.scatter(locations[:,0]*1000,locations[:,1]*1000,c='k',marker='.',s=10)
plt.xlabel('x location [mm]')
plt.ylabel('y location [mm]')
plt.title('Incoming rays %1.1f" off-axis' %(off_axis_as))

#plt.savefig('./figures/spot_diagram_%d.png'%(off_axis_as))


# In[11]:


### Optimize focal plane for certain field of view ###

# Calculate spot diagram for rays 0, 10, 20, 30" off-axis
# Find RMS diameter of each spot
# Plot average RMS vs focal plane location
# Should go like  \/  ? 


# In[93]:


focal_planes = np.linspace(-0.2,-0.1993,30)
angles = [0,1*60,2*60,3*60,4*60,5*60]

rms_vals = np.zeros(shape=(len(angles),len(focal_planes)))

start = time.time()

for i in range(0,len(focal_planes)):
    for j in range(0,len(angles)):
        # Off axis angle in arcsec, radian
        off_axis_as = angles[j]
        off_axis_rad = off_axis_as*4.8481368e-6
        #print('Off-axis angle: %f rad' %(off_axis_rad))

        ## Make an array of ray objects and propagate all of them through to the focus
        x_locs = np.linspace(-0.5,0.5,100,dtype='float')
        y_locs = np.linspace(-0.5,0.5,100,dtype='float')

        rays = []

        for x_loc in x_locs:
            for y_loc in y_locs:
                # Form the annulus of incoming rays by skipping the loop any time coords fall outside of the primary
                # or inside of the secondary
                if((x_loc**2+y_loc**2 > 0.5) or (x_loc**2+y_loc**2 < 0.09)):
                    continue
                else:
                    # Otherwise make a ray at x_loc, y_loc
                    rays.append(Ray(x_loc,y_loc,np.sin(off_axis_rad),0,-np.cos(off_axis_rad)))


        locations = np.zeros((len(rays),3))

        # Propagate each ray through the system and get the final location
        for k in range(0,len(rays)):
            ray = rays[k]
            locations[k] = ray.propagate(primary,secondary,a2,b2,focal=focal_planes[i])


        #Get rid of any of the rays we put up at [0,0,10] with a logical array
        rays_to_cull = np.zeros(len(rays),dtype=bool)
        for k in range(0,len(rays_to_cull)):
            if(locations[k][2] > 0):
                rays_to_cull[k] = True

        locations = locations[rays_to_cull==False]
        
        #Find RMS spot size
        mean = np.average(locations[:,0])
        rms_vals[j,i]=np.sqrt(np.average((locations[:,0]-mean)**2))

end = time.time()
print('elapsed time: %f' %(end-start))


# In[94]:


# Average spot size for each focal plane = average(:,0)
avg_rms = np.zeros(len(focal_planes))
for i in range(0,len(focal_planes)):
    avg_rms[i] = np.average(rms_vals[2::,i])

plt.figure(figsize=(8,8))
plt.xlabel(r'Focal plane distance from -0.200mm [mm]')
plt.ylabel(r'Average RMS spot size [$\mu$m]')
plt.scatter((focal_planes+0.2)*1000,rms_vals[:][0]*1e6,s=18,marker='+',label='0\'')
plt.scatter((focal_planes+0.2)*1000,rms_vals[:][1]*1e6,s=18,marker='o',label='1\'')
plt.scatter((focal_planes+0.2)*1000,rms_vals[:][2]*1e6,s=18,marker='x',label='2\'')
plt.scatter((focal_planes+0.2)*1000,rms_vals[:][3]*1e6,s=18,marker='v',label='3\'')
plt.scatter((focal_planes+0.2)*1000,rms_vals[:][4]*1e6,s=18,marker='^',label='4\'')
plt.scatter((focal_planes+0.2)*1000,rms_vals[:][5]*1e6,s=22,marker='2',label='5\'')
#plt.scatter((focal_planes+0.2)*1000,rms_vals[:][6]*1e6,s=22,marker='3',label='30\"')
plt.scatter((focal_planes+0.2)*1000,avg_rms[:]*1e6,s=22,marker='1',label='average (2-30\")')
plt.legend()


# In[126]:


plt.figure(figsize=(8,8))
plt.plot(focal_planes,rms_vals[3,:])
plt.xlabel('Focal plane z location')
plt.ylabel('RMS spot size')


# In[12]:


###########################################################
#Every thing below here is debugging & random calculations#
###########################################################
test_ray = Ray(-0.5,0,0,0,-1)
test_ray.propagate_to_primary()
test_ray.set_dir(primary.reflect(test_ray))
#print(test_ray.get_location())
#print(test_ray.get_dir())
#print(primary.normal(test_ray.get_location()))

r1 = np.asarray([0,0,-1])
r2 = test_ray.get_dir()
n = primary.normal(test_ray.get_location())
print(r2)

test_ray.propagate_to_secondary(a2,b2)
print(test_ray.get_location())

print(secondary.surface(-0.0890194,0))


off_ax = 5*60*4.8481368e-6
r_hat = np.asarray([np.sin(off_ax),0,-np.cos(off_ax)])
n_hat = primary.normal([0.5,0,0.5*1/12])
cos_theta = np.dot(-r_hat,n_hat)
outgoing = r_hat+2*cos_theta*n_hat
print(outgoing)
print(2*outgoing[0])
print(2*x_edge)


# In[25]:


s = secondary.surface(0,0) - primary.surface(0,0)


# In[31]:


q = secondary.surface(0,0)+0.2
print(q)
p = 3.2 - q
f_eff = 3*q/p
print(f_eff)


# In[28]:


print(secondary.surface(0,0))


# In[ ]:


off_axis_as = 5*60 # Off axis angle in arcsec
off_axis_rad = off_axis_as*4.8481368e-6 # off axis angle in radian
print(off_axis_rad)


# In[ ]:


r = np.asarray([np.sin(off_axis_rad),-np.cos(off_axis_rad)])
n = 1/np.linalg.norm([-1/6,1])*np.asarray([-1/6,1])
print(r)
print(n)
cos_theta = -np.dot(r,n)
print(cos_theta)
print(r+2*cos_theta*n)


# In[ ]:


ray = Ray(0.5,0,-0.32294816,0,0.94641666)
ray.propagate_to_secondary(a2,b2)


# In[ ]:


print(ray.get_location())


# In[ ]:


d=-x_edge/np.tan(off_axis_rad)


# In[ ]:


print(secondary.surface(-x_edge,0)-d)


# In[ ]:




