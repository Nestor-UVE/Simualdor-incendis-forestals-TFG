import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2

from scipy.ndimage import gaussian_filter

# on définit la résolution de calcul


class Vent2:
    def __init__(self, seed):
        self.dx=1
        self.dy=1
        self.dt=0.001
        self.subset=int(1/self.dx)

        # On définit la résolution du dessin
        self.dxd=.1
        self.dyd=.1

        self.rho = 1.247  # air 10° en kg/m3
        self.nu = 15.6 / (10 ** 6)  # air en m2/s https://fr.wikipedia.org/wiki/Viscosit%C3%A9_cin%C3%A9matique

        self.v0 = 20.0  # (km/h)
        self.v0 = self.v0 / 3.6
        self.p0 = 101300  # HPa

        self.lx = 100
        self.ly = 100
        # on crée les données
        self.sx = int(self.lx // self.dx)
        self.sy = int(self.ly // self.dy)

        self.sxd = int(self.lx // self.dxd)
        self.syd = int(self.ly // self.dyd)

        # On crée ensuite la scène
        self.X = np.linspace(0, self.lx, self.sx)
        self.Y = np.linspace(0, self.ly, self.sy)

        self.Vx = np.zeros((self.lx, self.ly))
        self.Vy = np.zeros((self.lx, self.ly))
        self.p=np.ones((self.sy,self.sx))*self.p0

        self.previous_random_u = 0
        self.previous_random_v = 0

        self.Vx_disturbance,self.Vy_disturbance  = self.generate_smoothed_random_field(seed[0])
        
        for _ in range(50):
            self.step(seed)

       


    def Subset(self, *X, s=2, c=0):
        res = []
        for x in X:
            if len(x.shape) == 2:
                t = x[c:-1-c, c:-1-c]
                res.append(t[::s, ::s])
            elif len(x.shape) == 1:
                t = x[c:-1-c]
                res.append(t[::s])
        if len(res) == 1:
            return res[0]
        return tuple(res)

    

    
    def Calcul_An_Bn(self, Vxn,Vyn,dx,dy):
        dVxn_dx=(Vxn[1:-1,2:]-Vxn[1:-1,:-2])/(2*dx)
        dVyn_dy=(Vyn[2:,1:-1]-Vyn[:-2,1:-1])/(2*dy)
        # dérivées croisées
        dVxn_dy=(Vxn[2:,1:-1]-Vxn[:-2,1:-1])/(2*dy)
        dVyn_dx=(Vyn[1:-1,2:]-Vyn[1:-1,:-2])/(2*dx)
        An=dVxn_dx+dVyn_dy
        Bn=dVxn_dx**2+dVyn_dy**2+2*dVxn_dy*dVyn_dx
        return An,Bn

    def Calcul_Pression(self, pn,Vxn,Vyn,dt,dx,dy):
        for i in range(10):
            # on commence par calculer les termes en P
            p0=(
                (
                (dx**2)*(pn[2:,1:-1]+pn[:-2,1:-1])+
                (dy**2)*(pn[1:-1,2:]+pn[1:-1,:-2])
                )
                /
                (2*(dx**2+dy**2))
            )
            facteur=(self.rho*dx**2*dy**2)/(2*(dx**2+dy**2))
            An,Bn=self.Calcul_An_Bn(Vxn,Vyn,dx,dy)
            pn[1:-1,1:-1]=facteur*(Bn-An/dt)+p0
            fbord=self.rho*dy**2/2
            dvy_y0=(Vyn[2:,0]-Vyn[:-2,0])/(2*dy)
            dvy_yf=(Vyn[2:,0]-Vyn[:-2,0])/(2*dy)
            self.p[1:-1,0]=fbord*(dvy_y0**2-dvy_y0/dt)+(pn[2:,0]+pn[:-2,0])/2  
            self.p[1:-1,-1]=fbord*(dvy_yf**2-dvy_yf/dt)+(pn[2:,-1]+pn[:-2,-1])/2
        return pn

    def Laplacien(self, Vx,Vy,dx,dy):
        Vx_x=(Vx[1:-1,:-2]-2*Vx[1:-1,1:-1]+Vx[1:-1,2:])/(dx**2)
        Vx_y=(Vx[:-2,1:-1]-2*Vx[1:-1,1:-1]+Vx[2:,1:-1])/(dy**2)
        Vy_x=(Vy[1:-1,:-2]-2*Vy[1:-1,1:-1]+Vy[1:-1,2:])/(dx**2)
        Vy_y=(Vy[:-2,1:-1]-2*Vy[1:-1,1:-1]+Vy[2:,1:-1])/(dy**2)
        return Vx_x+Vx_y,Vy_x+Vy_y
    
    def generate_smoothed_random_field(self, seed):
        """Generates a smoothed random field simulating wind speed."""
            # np.random.seed(seed)  # Ensures reproducibility
        # np.random.seed(seed)  # Setting seed for reproducibility
        # Vx = np.random.uniform(-20, 20, (self.lx, self.ly))
        # Vy = np.random.uniform(-20, 20, (self.lx, self.ly))
        # two random number between -1 and 1
        random = np.random.uniform(-1, 1, 2)
        random = np.clip(random, -0.5, 0.5)
        v_i = np.random.uniform(20, 30)
        Vx = np.ones((self.lx, self.ly)) * np.random.normal(random[1], 1, (self.lx, self.ly)) * v_i
        Vy = np.ones((self.lx, self.ly)) * np.random.normal(random[0], 1, (self.lx, self.ly)) * v_i
        #  plot the random field Vx, Vy

        # Suavitzeu els camps de vent per assegurar coherència
        Vx_smoothed = gaussian_filter(Vx, sigma=0.1)
        Vy_smoothed = gaussian_filter(Vy, sigma=0.1)

        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(Vx_smoothed, cmap='jet')
        # plt.colorbar()
        # plt.title('Vx')
        # plt.subplot(1, 2, 2)
        # plt.imshow(Vy_smoothed, cmap='jet')
        # plt.colorbar()
        # plt.title('Vy')
        # plt.show()
        return Vx_smoothed, Vy_smoothed
    
    def random_suavitzat(self, Vx, Vy, alpha=0.5):
        random_factor_u = np.random.uniform(low=-10, high=10, size=(self.lx, self.ly))
        random_factor_v = np.random.uniform(low=-10, high=10, size=(self.lx, self.ly))
        random_factor_u = random_factor_u * (1 - alpha) + self.previous_random_u * alpha
        random_factor_v = random_factor_v * (1 - alpha) + self.previous_random_v * alpha
        self.previous_random_u = random_factor_u
        self.previous_random_v = random_factor_v
        Vx += random_factor_u
        Vy += random_factor_v
        return Vx, Vy

    def attenuation(self, Vx, Vy, alpha=0.1):
        #take values outside the 64x64 center and attenuate them. that means that Vx and Vy will be divided by 10 for example
        border = 10
        Vx[:border] = Vx[:border] * alpha
        Vy[:border] = Vy[:border] * alpha
        Vx[-border:] = Vx[-border:] * alpha
        Vy[-border:] = Vy[-border:] * alpha
        Vx[:, :border] = Vx[:, :border] * alpha
        Vy[:, :border] = Vy[:, :border] * alpha
        Vx[:, -border:] = Vx[:, -border:] * alpha
        Vy[:, -border:] = Vy[:, -border:] * alpha
        return Vx, Vy

    def step(self, seed):
        
        self.Vx, self.Vy = self.random_suavitzat(self.Vx, self.Vy)
        self.Vx, self.Vy = self.attenuation(self.Vx, self.Vy)

        Xd,Yd,Vxd,Vyd=self.Subset(self.X/self.dxd,self.Y/self.dyd, self.Vx, self.Vy,s=4)

        plt.quiver(Xd,Yd,Vxd,Vyd,angles='xy')
        # plt.show()

        udiff = 1
        stepcount = 0
        Vel = 0
        V_dir = 0

        while 1:
            VxnI = self.Vx.copy()
            VynI = self.Vy.copy()
            pn=self.p.copy()
            Vxn=VxnI.copy()
            Vyn=VynI.copy()
            Vxn[1:-1,1:-1]=(VxnI[:-2,1:-1]+VxnI[2:,1:-1]+VxnI[1:-1,2:]+VxnI[1:-1,:-2])/4
            Vyn[1:-1,1:-1]=(VynI[:-2,1:-1]+VynI[2:,1:-1]+VynI[1:-1,2:]+VynI[1:-1,:-2])/4
            pn=self.Calcul_Pression(pn,Vxn,Vyn,self.dt,self.dx,self.dy)
            LapX,LapY=self.Laplacien(Vxn,Vyn,self.dx,self.dy)
            # dérivées 
            dVxn_dx=(Vxn[1:-1,2:]-Vxn[1:-1,:-2])/(2*self.dx)
            dVyn_dy=(Vyn[2:,1:-1]-Vyn[:-2,1:-1])/(2*self.dy)
            # dérivées croisées
            dVxn_dy=(Vxn[2:,1:-1]-Vxn[:-2,1:-1])/(2*self.dy)
            dVyn_dx=(Vyn[1:-1,2:]-Vyn[1:-1,:-2])/(2*self.dx)
            # dérivée pression
            dp_dx=(pn[1:-1,2:]-pn[1:-1,:-2])/(2*self.dx)
            dp_dy=(pn[2:,1:-1]-pn[:-2,1:-1])/(2*self.dy)
            # Equation e1 => Vitesse x
            self.Vx[1:-1, 1:-1] = (Vxn[1:-1, 1:-1]+self.dt*(
                                self.nu*LapX+
                                -Vxn[1:-1,1:-1]*dVxn_dx
                                -Vyn[1:-1,1:-1]*dVxn_dy
                                - (1/self.rho)*dp_dx
                            )
                            )
            self.Vx[1:-1,-1]=(Vxn[1:-1,-1]+self.dt*(
                self.nu*((self.Vx[:-2,-1]-2*self.Vx[1:-1,-1]+self.Vx[2:,-1])/(self.dy**2)) # On ne garde que le laplacien en X.
                - 0 # la variation en x est nulle
                -Vyn[1:-1,0]*(Vxn[2:,-1]-Vxn[:-2,-1])/(2*self.dy)
                - (1/self.rho)*0 # Variation p nulle en 0.
            ))
            # Equation e2 => Vitesse y
            self.Vy[1:-1, 1:-1] = (Vyn[1:-1, 1:-1]+self.dt*(
                                self.nu*LapY+
                                -Vxn[1:-1,1:-1]*dVyn_dx
                                -Vyn[1:-1,1:-1]*dVyn_dy
                                - (1/self.rho)*dp_dy
                            )
                            )    
            self.Vy[1:-1,-1]=(Vyn[1:-1,-1]+self.dt*(
                self.nu*((self.Vy[:-2,-1]-2*self.Vy[1:-1,-1]+self.Vy[2:,-1])/(self.dy**2)) # On ne garde que le laplacien en y.
                - 0 # la variation en x est nulle
                -Vyn[1:-1,0]*(Vyn[2:,-1]-Vyn[:-2,-1])/(2*self.dy)
                - (1/self.rho)*(pn[2:,-1]-pn[:-2,-1])/(2*self.dy) # Variation p nulle en 0.
            ))

            self.Vx=np.nan_to_num(self.Vx)
            VxnI=np.nan_to_num(VxnI)
            udiff = abs((np.sum(self.Vx) - np.sum(VxnI)) / np.sum(self.Vx))
            # print(stepcount)
            # print(udiff)

            Vel=np.sqrt(self.Vx**2+self.Vy**2) * 3.28084  # Convert to ft/s
            V_dir = np.arctan2(self.Vy, self.Vx)

            V_dir = np.degrees(V_dir)  # Convert to degrees
            

            Vel = Vel[(self.sx//2-32):(self.sx//2+32), (self.sy//2-32):(self.sy//2+32)]
            V_dir = V_dir[(self.sx//2-32):(self.sx//2+32), (self.sy//2-32):(self.sy//2+32)]
            
            if udiff<0.001:  #tret 1 zero
                break
            stepcount += 1
            
            # V_dir = 180
        return  Vel, V_dir, np.mean(Vel)

    # def plot(self, Vel, mean_V):

    #     # norm = matplotlib.colors.Normalize(vmin=0,vmax=100)

    #     fig = plt.figure(figsize = (11,11), dpi=100)
        
    #     plt.title(f"Wind Speed mean ={round(mean_V*1.1,1)}km/h")
    #     Xd,Yd,Vxd,Vyd,Vd=self.Subset(self.X,self.Y, self.Vx, self.Vy, Vel,s=self.subset)
    #     col = Vd*3.6
    #     lw=3*col/col.max()
    #     stream=plt.streamplot(Xd/self.dxd, Yd/self.dyd, Vxd, Vyd,color=col,density=[15,1],linewidth=lw,arrowsize=0.4)
    #     plt.colorbar(stream.lines,orientation='horizontal')
    #     plt.savefig("quiver_plot.png")
    #     plt.close()  # Close the plot to prevent displaying it

    #     # plt.show()

    def reset(self, seed):
        """Resets the wind fields."""
        self.sx = int(self.lx // self.dx)
        self.sy = int(self.ly // self.dy)

        self.sxd = int(self.lx // self.dxd)
        self.syd = int(self.ly // self.dyd)

        # On crée ensuite la scène
        self.X = np.linspace(0, self.lx, self.sx)
        self.Y = np.linspace(0, self.ly, self.sy)

        self.Vx = np.zeros((self.lx, self.ly))
        self.Vy = np.zeros((self.lx, self.ly))
        self.p=np.ones((self.sy,self.sx))*self.p0

        self.Vx_disturbance,self.Vy_disturbance  = self.generate_smoothed_random_field(seed[0])

        self.Vx += self.Vx_disturbance
        self.Vy += self.Vy_disturbance

        Vel=np.sqrt(self.Vx**2+self.Vy**2)
        V_dir = np.arctan2(self.Vy, self.Vx)

        #i want to take only 64 middle values in V array
        Vel = Vel[(self.sx//2-32):(self.sx//2+32), (self.sy//2-32):(self.sy//2+32)]
        V_dir = V_dir[(self.sx//2-32):(self.sx//2+32), (self.sy//2-32):(self.sy//2+32)]

        for _ in range(50):
            self.step(seed)
        return Vel, V_dir


if __name__ == "__main__":
    vent = Vent2()
    Vel, V_dir, mean_V = vent.step()
    vent.plot(Vel, mean_V)

