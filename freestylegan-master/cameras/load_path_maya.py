# script to load a lookat file into Autodesk Maya (run in script editor)

import maya.cmds as cmds

#========================================

res = 1024

lookatFile = "path/to/camera.lookat"
    
#========================================

def loadParams():
    
    #-----------------------------    
    def parse3DAttributeString(s):
        a = s.replace('=', ',').split(",")[1:]
        return [float(x) for x in a]
        
    #-----------------------------

    transList = []
    rotList = []
    
    file = open(lookatFile, 'r')
    
    for line in file:    

        line = line.split()
        pos = parse3DAttributeString(line[2])
        target = parse3DAttributeString(line[4])
        up = parse3DAttributeString(line[6])
        fov = float(line[8].split("=")[1])
            
        cam = cmds.camera()
        camTrans = cam[0]
        camShape = cam[1]
        
        cmds.viewPlace(camShape, p=True, eye=pos, la=target, up=up)
        
        transl = cmds.xform(camTrans, q=True, ws=True, t=True)
        rot = cmds.xform(camTrans, q=True, ws=True, ro=True)
        transList.append(transl)
        rotList.append(rot)
        
        cmds.delete(cam)
    
    return transList, rotList, fov

#========================================

def main():
       
    # parse path
    transList, rotList, fov = loadParams()  
    
    #create camera    
    cam = cmds.camera()
    camTrans = cam[0]
    camShape = cam[1]
    cmds.camera(camShape, e=True, hfa=res/100.)
    cmds.camera(camShape, e=True, vfa=res/100.)
    cmds.camera(camShape, e=True, vfv=fov)
    
    # animate camera
    for idx, (trans, rot) in enumerate(zip(transList, rotList)):    
        
        cmds.currentTime(idx+1)
        
        cmds.setKeyframe(camTrans, v=trans[0], at='translateX')
        cmds.setKeyframe(camTrans, v=trans[1], at='translateY')
        cmds.setKeyframe(camTrans, v=trans[2], at='translateZ')
        cmds.setKeyframe(camTrans, v=rot[0], at='rotateX')
        cmds.setKeyframe(camTrans, v=rot[1], at='rotateY')
        cmds.setKeyframe(camTrans, v=rot[2], at='rotateZ')


#========================================

main()

