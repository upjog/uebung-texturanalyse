import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import cv2

# Initial parameters
ksize = 5
sigma = 3
theta = np.pi / 4
lambd = np.pi / 4
gamma = 0.5
phi = 0

# Function to update the plot based on slider values
def update(val):
    global ksize, sigma, theta, lambd, gamma, phi

    ksize = slider_ksize.val
    sigma = slider_sigma.val
    theta = slider_theta.val
    lambd = slider_lambd.val
    gamma = slider_gamma.val
    phi = slider_phi.val

    # Generate Gabor kernel with current slider values
    kernel = cv2.getGaborKernel((int(ksize), int(ksize)), sigma, theta, lambd, gamma, phi, ktype=cv2.CV_32F)

    # Update the plot
    ax.imshow(kernel)
    fig.canvas.draw_idle()

# Functions to adjust sliders with the arrow buttons
def adjust_slider(val, slider, step):
    slider.set_val(slider.val + step)
    update(val)

increase_ksize = lambda val: adjust_slider(val, slider_ksize, 2)
decrease_ksize = lambda val: adjust_slider(val, slider_ksize, -2)

increase_sigma = lambda val: adjust_slider(val, slider_sigma, 0.1)
decrease_sigma = lambda val: adjust_slider(val, slider_sigma, -0.1)

increase_theta = lambda val: adjust_slider(val, slider_theta, np.pi / 100)
decrease_theta = lambda val: adjust_slider(val, slider_theta, -np.pi / 100)

increase_lambd = lambda val: adjust_slider(val, slider_lambd, np.pi / 100)
decrease_lambd = lambda val: adjust_slider(val, slider_lambd, -np.pi / 100)

increase_gamma = lambda val: adjust_slider(val, slider_gamma, 0.05)
decrease_gamma = lambda val: adjust_slider(val, slider_gamma, -0.05)

increase_phi = lambda val: adjust_slider(val, slider_phi, np.pi / 100)
decrease_phi = lambda val: adjust_slider(val, slider_phi, -np.pi / 100)


# Set up the plot with more space on the right for the sliders
fig, ax = plt.subplots(figsize=(6, 6))
plt.subplots_adjust(left=0.1, right=0.5, top=0.9, bottom=0.2)

# Initial plot with default values
kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, phi, ktype=cv2.CV_32F)
ax.imshow(kernel)
ax.axis('off')

# Add sliders to the right of the image
ax_ksize = plt.axes([0.7, 0.75, 0.15, 0.03], facecolor='lightgoldenrodyellow')
slider_ksize = Slider(ax_ksize, 'Ksize', 3, 100, valinit=ksize, valstep=2)

ax_sigma = plt.axes([0.7, 0.65, 0.15, 0.03], facecolor='lightgoldenrodyellow')
slider_sigma = Slider(ax_sigma, 'Sigma', 1, 10, valinit=sigma)

ax_theta = plt.axes([0.7, 0.55, 0.15, 0.03], facecolor='lightgoldenrodyellow')
slider_theta = Slider(ax_theta, 'Theta', 0, 2*np.pi, valinit=theta)

ax_lambd = plt.axes([0.7, 0.45, 0.15, 0.03], facecolor='lightgoldenrodyellow')
slider_lambd = Slider(ax_lambd, 'Lambda', 0, 2*np.pi, valinit=lambd)

ax_gamma = plt.axes([0.7, 0.35, 0.15, 0.03], facecolor='lightgoldenrodyellow')
slider_gamma = Slider(ax_gamma, 'Gamma', 0.1, 5, valinit=gamma)

ax_phi = plt.axes([0.7, 0.25, 0.15, 0.03], facecolor='lightgoldenrodyellow')
slider_phi = Slider(ax_phi, 'Phi', 0, 2*np.pi, valinit=phi)

# Add arrow buttons
ax_left_ksize = plt.axes([0.62, 0.75, 0.02, 0.03], facecolor='lightgoldenrodyellow')
btn_left_ksize = Button(ax_left_ksize, '<', color='lightgoldenrodyellow')
btn_left_ksize.on_clicked(decrease_ksize)

ax_right_ksize = plt.axes([0.89, 0.75, 0.02, 0.03], facecolor='lightgoldenrodyellow')
btn_right_ksize = Button(ax_right_ksize, '>', color='lightgoldenrodyellow')
btn_right_ksize.on_clicked(increase_ksize)

ax_left_sigma = plt.axes([0.62, 0.65, 0.02, 0.03], facecolor='lightgoldenrodyellow')
btn_left_sigma = Button(ax_left_sigma, '<', color='lightgoldenrodyellow')
btn_left_sigma.on_clicked(decrease_sigma)

ax_right_sigma = plt.axes([0.89, 0.65, 0.02, 0.03], facecolor='lightgoldenrodyellow')
btn_right_sigma = Button(ax_right_sigma, '>', color='lightgoldenrodyellow')
btn_right_sigma.on_clicked(increase_sigma)

ax_left_theta = plt.axes([0.62, 0.55, 0.02, 0.03], facecolor='lightgoldenrodyellow')
btn_left_theta = Button(ax_left_theta, '<', color='lightgoldenrodyellow')
btn_left_theta.on_clicked(decrease_theta)

ax_right_theta = plt.axes([0.89, 0.55, 0.02, 0.03], facecolor='lightgoldenrodyellow')
btn_right_theta = Button(ax_right_theta, '>', color='lightgoldenrodyellow')
btn_right_theta.on_clicked(increase_theta)

ax_left_lambd = plt.axes([0.62, 0.45, 0.02, 0.03], facecolor='lightgoldenrodyellow')
btn_left_lambd = Button(ax_left_lambd, '<', color='lightgoldenrodyellow')
btn_left_lambd.on_clicked(decrease_lambd)

ax_right_lambd = plt.axes([0.89, 0.45, 0.02, 0.03], facecolor='lightgoldenrodyellow')
btn_right_lambd = Button(ax_right_lambd, '>', color='lightgoldenrodyellow')
btn_right_lambd.on_clicked(increase_lambd)

ax_left_gamma = plt.axes([0.62, 0.35, 0.02, 0.03], facecolor='lightgoldenrodyellow')
btn_left_gamma = Button(ax_left_gamma, '<', color='lightgoldenrodyellow')
btn_left_gamma.on_clicked(decrease_gamma)

ax_right_gamma = plt.axes([0.89, 0.35, 0.02, 0.03], facecolor='lightgoldenrodyellow')
btn_right_gamma = Button(ax_right_gamma, '>', color='lightgoldenrodyellow')
btn_right_gamma.on_clicked(increase_gamma)

ax_left_phi = plt.axes([0.62, 0.25, 0.02, 0.03], facecolor='lightgoldenrodyellow')
btn_left_phi = Button(ax_left_phi, '<', color='lightgoldenrodyellow')
btn_left_phi.on_clicked(decrease_phi)

ax_right_phi = plt.axes([0.89, 0.25, 0.02, 0.03], facecolor='lightgoldenrodyellow')
btn_right_phi = Button(ax_right_phi, '>', color='lightgoldenrodyellow')
btn_right_phi.on_clicked(increase_phi)

# Update function for sliders
slider_ksize.on_changed(update)
slider_sigma.on_changed(update)
slider_theta.on_changed(update)
slider_lambd.on_changed(update)
slider_gamma.on_changed(update)
slider_phi.on_changed(update)

# Show the plot
plt.show()
