
import math
import argparse

import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.animation as animation

from svgpathtools import svg2paths

def poly_shift_x(poly,shift_val):
    ''' Shift poly to the right by shift_val.
    args:
        poly: c_n * x^n + ... c_2 * x^2 + x_1 * x + c_0
        shift_val: value by which to shift poly in the x direction
    
    return:
       poly_shifted: polynomial curve poly shifted to the right by shift_val
    
         \          /  -->
          \        /   -->
           \      /    -->
            \____/     -->
    ''' 
    poly_shifted = np.poly1d([0])

    for i_n, coeff_n in enumerate(poly.c):
        exponent = poly.o - i_n

        shifted_x_term = np.poly1d([1,-shift_val])**exponent
        poly_shifted = poly_shifted+coeff_n*shifted_x_term

    return poly_shifted

def poly_rescale_x(poly,rescale_const):
    ''' Rescale poly horizontally by the rescale_constant.
    args:
        poly: c_n * x^n + ... c_2 * x^2 + x_1 * x + c_0
        rescale_const: value by which to rescale poly in the x direction
    
    return:
       poly_rescaled: polynomial curve poly rescaled horizontally by rescale_const
    '''
    poly_rescaled_coeffs = []

    for i_n, coeff_n in enumerate(poly.c):
        exponent = poly.o - i_n

        coeff_n_new = coeff_n*(1.0/(rescale_const**exponent))
        poly_rescaled_coeffs.append(coeff_n_new)

    poly_rescaled = np.poly1d(poly_rescaled_coeffs)

    return poly_rescaled

def poly_conj(poly):
    ''' Giuven a polynomial with complex coefficients, return another polynomial
    with coefficients that are the conjugate of the input polynomial's complex 
    coefficients.

    If this polynomial represents 2d motion over time, this function has the
    effect of flipping the resulting motion vertically on the 2d plane.
        
        Ex:
            poly.c = [ a+bi , c+di ]

        poly_flip_ud = 
    '''

    poly_conj_coeffs = []

    for coeff_n in poly.c:
        poly_conj_coeffs.append(coeff_n.conj())

    poly_conj = np.poly1d(poly_conj_coeffs)

    return poly_conj

def poly_paths_from_svg(n, svg_path):
    ''' Convert the contents of svg_filename into a series of polynomials, one
        for each svg path segment, where the segments piecewise span 
        0 < t < 2*pi.
        
        Ex:
        seg1 = poly1d   0  < t1
        seg2 = poly1d   t1 < t2
        seg3 = poly1d   t2 < 2*pi

        poly_segments = [s1, s2, s3]
    '''

    num_segments = len(svg_path)
    dt_segment = 2*math.pi/num_segments

    x_i_contrib = np.zeros([num_segments,2*n+1],dtype=np.complex128)

    poly_segments = []

    for segment_idx,segment in enumerate(svg_path):

        t_start = segment_idx*dt_segment
        t_end = (segment_idx+1)*dt_segment

        seg_poly = segment.poly()
        seg_poly_rescaled = poly_rescale_x(seg_poly,dt_segment)
        seg_poly_shift_x = poly_shift_x(seg_poly_rescaled,t_start)
        seg_poly_flip_ud = poly_conj(seg_poly_shift_x)
        poly_segments.append(seg_poly_flip_ud)

    return poly_segments


def find_epicycles(n, poly_segments):

    num_segments = len(poly_segments)
    dt_segment = 2*math.pi/num_segments

    x_i_contrib = np.zeros([num_segments,2*n+1],dtype=np.complex128)

    for segment_idx,poly_segment in enumerate(poly_segments):

        t_start = segment_idx*dt_segment
        t_end = (segment_idx+1)*dt_segment
        n_t = 200                                       # MAY NEED TO MODIFY
        t_segment_pts = np.linspace(t_start,t_end,n_t)

        for n_idx in range(2*n+1):
            i = n_idx - n
            pts_to_integrate = poly_segment(t_segment_pts)*np.exp(-1j*i*t_segment_pts)
            x_i_contrib[segment_idx][n_idx] = (1/(2.0*math.pi))*integrate.trapz(pts_to_integrate,x=t_segment_pts)

    x_elems = np.sum(x_i_contrib,axis=0)

    return x_elems

def update_animation(k,shapes_epicycles,line_objects,circles_objects,trajectory_objects,t_n_samples):

    k_first_cycle = int(math.ceil(t_n_samples/2.0))

    for shape_idx,shape_epicycle in enumerate(shapes_epicycles):

        for i in range(1,len(shape_epicycle['f_t'])):
            seg_1_x = shape_epicycle['f_t'][i-1][k].real
            seg_1_y = shape_epicycle['f_t'][i-1][k].imag
            
            seg_2_x = shape_epicycle['f_t'][i][k].real
            seg_2_y = shape_epicycle['f_t'][i][k].imag    

            line_objects[shape_idx][i].set_data([seg_1_x,seg_2_x], [seg_1_y,seg_2_y])
            
            circles_objects[shape_idx][i].center = (seg_1_x,seg_1_y)

            # Create drawing over time in first half of animation, delete over time in second half of animation
            if k < k_first_cycle:
                trajectory_objects[shape_idx][0].set_data(shape_epicycle['f_t'][-1][:k].real,shape_epicycle['f_t'][-1][:k].imag)
            else:
                k_delete=k-k_first_cycle
                trajectory_objects[shape_idx][0].set_data(shape_epicycle['f_t'][-1][k_delete:k_first_cycle].real,shape_epicycle['f_t'][-1][k_delete:k_first_cycle].imag)

    return line_objects, circles_objects, trajectory_objects

def animate(x_elem_shapes):

    num_paths = len(x_elem_shapes)

    n = len(x_elem_shapes[0])

    # Create time vector
    num_cycles = 2
    t_final = 2*math.pi*num_cycles
    t_n_samples = 100                           # MAY NEED TO CHANGE
    t = np.linspace(0.0,t_final,t_n_samples)   
    
    # Create epicycles array with f(t) info for each circular path in each shape
    shapes_epicycles = []

    dtype_obj = np.dtype([('x','complex'),('r','float'),('k','int'),('f_t','object')])

    for x_elems_shape in x_elem_shapes:

        epicycles = np.array([],dtype=dtype_obj)

        for idx,x_elem in enumerate(x_elems_shape):
            k = idx - math.floor(len(x_elems_shape)/2)
            r = np.absolute(x_elem)
            f_t = x_elem*np.exp(1j*k*t)
            epicycles = np.append(epicycles, np.array([(x_elem,r,k,f_t)],dtype=dtype_obj))

        epicycle_static = epicycles[math.floor(len(x_elems_shape)/2)]      # Extract element with zero rotation
        epicycles = np.delete(epicycles,math.floor(len(x_elems_shape)/2))  # Remove element with zero rotation
        epicycles.sort(axis=0,order=['r'])
        epicycles = epicycles[::-1]
        epicycles = np.insert(epicycles, 0, epicycle_static)

        # Find f(t) in inertial coordinate frame
        f_t_epicycles = np.cumsum(epicycles['f_t'])

        for i,f_t_inertial in enumerate(f_t_epicycles):
            epicycles[i]['f_t']=f_t_inertial

        shapes_epicycles.append(epicycles)

    # Find max radius for setting axis limits
    epicycle_x_max = shapes_epicycles[0][-1]['f_t'][0].real
    epicycle_x_min = shapes_epicycles[0][-1]['f_t'][0].real
    epicycle_y_max = shapes_epicycles[0][-1]['f_t'][0].imag
    epicycle_y_min = shapes_epicycles[0][-1]['f_t'][0].imag

    for epicycles in shapes_epicycles:
        for pt in epicycles[-1]['f_t']:
            epicycle_x_max = max(epicycle_x_max,pt.real)
            epicycle_x_min = min(epicycle_x_min,pt.real)
            epicycle_y_max = max(epicycle_y_max,pt.imag)
            epicycle_y_min = min(epicycle_y_min,pt.imag)

    # Plot final trajectories
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(1,1,1)

    for shape_epicycle in shapes_epicycles:
        ax2.plot(shape_epicycle['f_t'][-1][:].real,shape_epicycle['f_t'][-1][:].imag)
    
    ax2.axis('equal')

    # Plot initial animation setup
    fig = plt.figure(1)
    ax = fig.add_subplot(1,1,1)
    ax.axis('equal')
    ax.patch.set_facecolor((1.0, 1.0, 1.0,1.0))
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    trajectory_color = (0.4,0.4,0.4,0.8)
    # trajectory_color = 'r'
    line_color = (0.8, 0.8, 1.0,0.5)
    # line_color = 'b'
    circle_color = (0.9607843137254902, 0.9607843137254902, 0.8627450980392157,0.8)
    # circle_color = 'c'

    trajectory_objects = []
    line_objects = []
    circles_objects = []

    for k,shape_epicycle in enumerate(shapes_epicycles):
        trajectory_objects.append(ax.plot(shape_epicycle[-1]['f_t'][0].real,shape_epicycle[-1]['f_t'][0].imag, color=trajectory_color))
        line_objects.append(ax.plot([shape_epicycle[0]['f_t'][0].real,shape_epicycle[0]['f_t'][1].real], [shape_epicycle[0]['f_t'][0].imag,shape_epicycle[0]['f_t'][1].imag], color=line_color))
        circles_objects.append([ax.add_patch(patches.Circle((shape_epicycle['f_t'][0][0].real,shape_epicycle['f_t'][0][0].imag), radius=epicycles['r'][1], fill=False, color=circle_color))])

        # print 'point 1 = '+str((shape_epicycle['f_t'][0][0].real,shape_epicycle['f_t'][0][0].imag))
        # print 'circle center = '+str(circles_objects[0][0].center)

        for i in range(1,len(shape_epicycle['f_t'])):
            seg_1_x = shape_epicycle['f_t'][i-1][0].real
            seg_1_y = shape_epicycle['f_t'][i-1][0].imag
            
            seg_2_x = shape_epicycle['f_t'][i][0].real
            seg_2_y = shape_epicycle['f_t'][i][0].imag    

            line_objects[k].append(ax.plot([seg_1_x,seg_2_x], [seg_1_y,seg_2_y], color=line_color)[0])
            circles_objects[k].append(ax.add_patch(patches.Circle((seg_2_x,seg_2_y), radius=epicycles['r'][i], fill=False, color=circle_color)))

    ax.axis([epicycle_x_min-200,epicycle_x_max+200,epicycle_y_min-200,epicycle_y_max+200])

    animation_tics = np.array(range(len(shapes_epicycles[0]['f_t'][0])))
    ani = animation.FuncAnimation(fig, update_animation,frames=animation_tics,fargs=(shapes_epicycles,line_objects,circles_objects,trajectory_objects,t_n_samples))
    # # ani.save('test.gif', writer='imagemagick', fps=24, dpi=75,savefig_kwargs={'transparent': True, 'facecolor': 'none'},extra_args={'transparent': True})
    # # ani.save('test.gif', writer='imagemagick', fps=24, dpi=75)

    plt.show()

def main(args):

    svg_paths,svg_attributes = svg2paths(args.filename)

    x_elem_shapes = []

    for path in svg_paths:
        poly_segments = poly_paths_from_svg(args.n, path)
        x_elem_shapes.append(find_epicycles(args.n, poly_segments))
    
    print x_elem_shapes

    animate(x_elem_shapes)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, required=True, help='Name of the svg file to create epicycles for.')
    parser.add_argument('--n', type=int, required=True, help='Number of epicycles to approximate.')
    args = parser.parse_args()

    main(args)