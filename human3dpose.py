import numpy as np 
import itertools
import eulerangles as EL
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
from sklearn.cross_validation import KFold
import pickle as pkl
def transform3D2D(M,rt):
	fx = 1000.
	fy = 1000.
	n = len(M)
	Rmat = EL.euler2mat(rt[0],rt[1],rt[2])
	MR = M.dot(Rmat)
	ts = np.tile(rt[3::],(n,1))
	p3d = MR+ts
	px = p3d[:,0]/p3d[:,2]#it is normalized by the fx and fy as described in page 12. 
	py = p3d[:,1]/p3d[:,2]
	p2d = np.vstack((px,py)).reshape((len(px)*2,1),order='F')
	p2d += np.random.normal(0,4.0,len(p2d)).reshape((len(p2d),1))/fx#add \sigma=4 white noise
	return p2d
def generate_pose(M,rmin,rmax,rstep,tmin,tmax,tstep):
	angle = np.radians(np.arange(rmin,rmax+rstep,rstep))
	t = np.arange(tmin,tmax+tstep,tstep)/1000.# in meters
	iterables = [angle,angle,angle,t,t,t+2]#z axis is at 2000mm
	rts = [t for t in itertools.product(*iterables)]
	return np.array(rts)
def initialize_pose(M,num):
	p2d0 = transform3D2D(M,[0.,0.,0.,0.,0.,2.])
	p2d0s = np.tile(p2d0.T,(num,1))
	rt0s = np.zeros((num,6))
	rt0s[:,5] = 2.
	return(p2d0s,rt0s)
def cascade_train(rts,rt0s,p2ds,p2d0s):
	cascades = list()
	for k in xrange(10):
		# kf = KFold(len(p2ds), n_folds=10)
		# rg = RidgeCV(alphas=[1.,10.,100.,1000.],cv=kf)
		rg = Ridge(alpha=70.)
		xs = p2ds - p2d0s
		ys = rts - rt0s
		print 'current error',np.mean(np.abs(ys),axis=0)
		rg.fit(xs,ys)
		#print "alpha is",rg.alpha_
		updatey = rg.predict(xs)
		rt0s += updatey
		p2d0s = [transform3D2D(M,rt) for rt in rt0s]
		p2d0s = np.array(p2d0s)[:,:,0]
		cascades.append(rg)
		print "the "+str(k)+"th level is trained"
	output  = open('cascadesHUMAN70.pkl','wb')
	pkl.dump(cascades,output)
	output.close()
	print 'Cascaded saved'
	return cascades
def cascade_test(cascades,p2ds,p2d0s,rt0s):
	for cascade in cascades:
		xs = p2ds - p2d0s
		rt0s += cascade.predict(xs)
		p2d0s = [transform3D2D(M,rt) for rt in rt0s]
		p2d0s = np.array(p2d0s)[:,:,0]
	return rt0s


need_to_train = True
M = np.loadtxt('3Dhuman.txt')# this is in meter
'''cascades training'''

'''
if need_to_train:
	rts = generate_pose(M,-30.,30.,10.,-400.,400.,200.)
	p2ds = [transform3D2D(M,rt) for rt in rts]
	p2ds = np.array(p2ds)[:,:,0] 
	p2d0s,rt0s = initialize_pose(M,len(p2ds))
	cascades = cascade_train(rts,rt0s,p2ds,p2d0s)
else:
	f = open('cascadesHUMAN70.pkl')
	cascades = pkl.load(f)
#cascades testing
test_rts = generate_pose(M,-30.,30.,7.,-400.,400.,170.)
test_p2ds = [transform3D2D(M,rt) for rt in test_rts]
test_p2ds = np.array(test_p2ds)[:,:,0] 
test_p2d0s,test_rt0s = initialize_pose(M,len(test_p2ds))
rts_result = cascade_test(cascades,test_p2ds,test_p2d0s,test_rt0s)
merr = np.mean(np.abs(test_rts - rts_result),axis=0)
print 'test error',merr
np.savetxt("error.txt",merr)
# f2 = open('test70_angle20.pkl','wb')
# pkl.dump(test_rts,f2)
# pkl.dump(rts_result,f2)
# f2.close()
'''

