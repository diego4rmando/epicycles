import math
import argparse

import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.animation as animation
import csv

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

        # Create drawing over time in first half of animation, delete over time in second half of animation
        if k < k_first_cycle:
            trajectory_objects[shape_idx][0].set_data(shape_epicycle['f_t'][-1][:k+1].real,shape_epicycle['f_t'][-1][:k+1].imag)
        else:
            k_delete=k-k_first_cycle
            trajectory_objects[shape_idx][0].set_data(shape_epicycle['f_t'][-1][k_delete:k_first_cycle].real,shape_epicycle['f_t'][-1][k_delete:k_first_cycle].imag)

        for i, shape_epicycle_f_t in enumerate(shape_epicycle['f_t']):

            seg_1_x = shape_epicycle['f_t'][i-1][k].real
            seg_1_y = shape_epicycle['f_t'][i-1][k].imag
            
            seg_2_x = shape_epicycle['f_t'][i][k].real
            seg_2_y = shape_epicycle['f_t'][i][k].imag    

            if i >= 1:
                line_objects[shape_idx][i-1].set_data([seg_1_x,seg_2_x], [seg_1_y,seg_2_y])
                
                circles_objects[shape_idx][i-1].center = (seg_1_x,seg_1_y)

    return line_objects, circles_objects, trajectory_objects

def animate(x_elem_shapes, svg_attributes, gif_filename, csv_filename, color_theme):

    num_paths = len(x_elem_shapes)

    n = len(x_elem_shapes[0])

    # Create time vector
    num_cycles = 2
    t_final = 2*math.pi*num_cycles
    t_n_samples = 300                           # MAY NEED TO CHANGE
    t = np.linspace(0.0,t_final,t_n_samples)   
    
    # Create epicycles array with f(t) info for each circular path in each shape
    shapes_epicycles = []

    dtype_obj = np.dtype([('x','complex'),('r','float'),('k','int'),('f_t','object')])

    for x_elems_shape, attributes_list in zip(x_elem_shapes,svg_attributes):

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

    # # Plot final trajectories

    # Plot initial animation setup
    fig = plt.figure()
    fig.patch.set_facecolor((0.96, 0.96, 1.0,1.0))
    ax = fig.add_subplot(1,1,1)
    ax.axis('equal')
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    if color_theme == 'electric_blue':
        trajectory_color = (0.55,0.64,0.76,0.8)
        line_color = (0.2, 0.52, 0.99,0.4)
        circle_color = (0.8, 0.87, 0.99,0.4)
    else:
        trajectory_color = (0.5,0.5,0.5,0.5)
        line_color = (0.75, 0.75, 0.75,0.5)
        circle_color = (0.9, 0.9, 0.9,0.5)

    trajectory_objects = []
    line_objects = []
    circles_objects = []

    for shape_idx,(shape_epicycle, attributes) in enumerate(zip(shapes_epicycles, svg_attributes)):

        shape_color = attributes.get('stroke',trajectory_color)
        trajectory_objects.append(ax.plot(shape_epicycle[-1]['f_t'][0].real,shape_epicycle[-1]['f_t'][0].imag, color=shape_color))

        for i, shape_epicycle_f_t in enumerate(shape_epicycle['f_t']):
            seg_1_x = shape_epicycle['f_t'][i-1][0].real
            seg_1_y = shape_epicycle['f_t'][i-1][0].imag
        
            seg_2_x = shape_epicycle['f_t'][i][0].real
            seg_2_y = shape_epicycle['f_t'][i][0].imag 

            if i == 1:
                line_objects.append(ax.plot([seg_1_x,seg_2_x], [seg_1_y,seg_2_y], color=line_color))
                circles_objects.append([ax.add_patch(patches.Circle((seg_2_x,seg_2_y), radius=shape_epicycle['r'][i], fill=False, color=circle_color))])
            elif i>1:  
                line_objects[shape_idx].append(ax.plot([seg_1_x,seg_2_x], [seg_1_y,seg_2_y], color=line_color)[0])
                circles_objects[shape_idx].append(ax.add_patch(patches.Circle((seg_2_x,seg_2_y), radius=shape_epicycle['r'][i], fill=False, color=circle_color)))

    ax.axis([epicycle_x_min,epicycle_x_max,epicycle_y_min,epicycle_y_max])

    animation_tics = np.array(range(len(shapes_epicycles[0]['f_t'][0])))
    ani = animation.FuncAnimation(fig, update_animation,frames=animation_tics,fargs=(shapes_epicycles,line_objects,circles_objects,trajectory_objects,t_n_samples))
    
    if gif_filename is not None:
        if (gif_filename[-4:]) != '.gif':
            gif_filename = gif_filename + '.gif'
        ani.save(gif_filename, writer='imagemagick', fps=6, dpi=200)

    if csv_filename is not None:
        if (csv_filename[-4:]) != '.csv':
            csv_filename = csv_filename + '.csv'
        with open(csv_filename, mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow(['shape_idx', 'k', 'x'])
            for shape_idx, shape_info in enumerate(shapes_epicycles):
                for epicycle_info in shape_info:
                    csv_writer.writerow([shape_idx, epicycle_info['k'], epicycle_info['x']])
            csv_file.close()

    plt.show()

def main(args):

    svg_paths,svg_attributes = svg2paths(args.filename)

    x_elem_shapes = []

    for path in svg_paths:
        poly_segments = poly_paths_from_svg(args.n, path)
        x_elem_shapes.append(find_epicycles(args.n, poly_segments))

    animate(x_elem_shapes,svg_attributes,args.gif_filename, args.csv_filename, args.color_theme)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help='Name of the svg file to create epicycles for.')
    parser.add_argument('--n', type=int, required=True, help='Number of epicycles to approximate.')
    parser.add_argument('--gif_filename',type=str, required=False, default=None)
    parser.add_argument('--csv_filename',type=str, required=False, default=None)    
    parser.add_argument('--color_theme',type=str, required=False, default=None)

    args = parser.parse_args()

    main(args)