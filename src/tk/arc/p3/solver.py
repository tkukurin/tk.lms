"""Solvers adapted from [0]

[0]: https://github.com/michaelhodel/re-arc
"""
from .dsl import *
from .const import *


def solve_67a3c6ac(I):
    O = vmirror(I)
    return O


def solve_68b16354(I):
    O = hmirror(I)
    return O


def solve_74dd1130(I):
    O = dmirror(I)
    return O


def solve_3c9b0459(I):
    O = rot180(I)
    return O


def solve_6150a2bd(I):
    O = rot180(I)
    return O


def solve_9172f3a0(I):
    O = upscale(I, 3)
    return O


def solve_9dfd6313(I):
    O = dmirror(I)
    return O


def solve_a416b8f3(I):
    O = hconcat(I, I)
    return O


def solve_b1948b0a(I):
    O = replace(I, 6, 2)
    return O


def solve_c59eb873(I):
    O = upscale(I, 2)
    return O


def solve_c8f0f002(I):
    O = replace(I, 7, 5)
    return O


def solve_d10ecb37(I):
    O = crop(I, ORIGIN, G2x2)
    return O


def solve_d511f180(I):
    O = switch(I, 5, 8)
    return O


def solve_ed36ccf7(I):
    O = rot270(I)
    return O


def solve_4c4377d9(I):
    x1 = hmirror(I)
    O = vconcat(x1, I)
    return O


def solve_6d0aefbc(I):
    x1 = vmirror(I)
    O = hconcat(I, x1)
    return O


def solve_6fa7a44f(I):
    x1 = hmirror(I)
    O = vconcat(I, x1)
    return O


def solve_5614dbcf(I):
    x1 = replace(I, 5, 0)
    O = downscale(x1, 3)
    return O


def solve_5bd6f4ac(I):
    x1 = tojvec(6)
    O = crop(I, x1, G3x3)
    return O


def solve_5582e5ca(I):
    x1 = mostcolor(I)
    O = canvas(x1, G3x3)
    return O


def solve_8be77c9e(I):
    x1 = hmirror(I)
    O = vconcat(I, x1)
    return O


def solve_c9e6f938(I):
    x1 = vmirror(I)
    O = hconcat(I, x1)
    return O


def solve_2dee498d(I):
    x1 = hsplit(I, 3)
    O = first(x1)
    return O


def solve_1cf80156(I):
    x1 = objects(I, T, T, T)
    x2 = first(x1)
    O = subgrid(x2, I)
    return O


def solve_32597951(I):
    x1 = ofcolor(I, 8)
    x2 = delta(x1)
    O = fill(I, 3, x2)
    return O


def solve_25ff71a9(I):
    x1 = objects(I, T, T, T)
    x2 = first(x1)
    O = move(I, x2, DOWN)
    return O


def solve_0b148d64(I):
    x1 = partition(I)
    x2 = argmin(x1, size)
    O = subgrid(x2, I)
    return O


def solve_1f85a75f(I):
    x1 = objects(I, T, T, T)
    x2 = argmax(x1, size)
    O = subgrid(x2, I)
    return O


def solve_23b5c85d(I):
    x1 = objects(I, T, T, T)
    x2 = argmin(x1, size)
    O = subgrid(x2, I)
    return O


def solve_9ecd008a(I):
    x1 = vmirror(I)
    x2 = ofcolor(I, 0)
    O = subgrid(x2, x1)
    return O


def solve_ac0a08a4(I):
    x1 = colorcount(I, 0)
    x2 = subtract(9, x1)
    O = upscale(I, x2)
    return O


def solve_be94b721(I):
    x1 = objects(I, T, F, T)
    x2 = argmax(x1, size)
    O = subgrid(x2, I)
    return O


def solve_c909285e(I):
    x1 = leastcolor(I)
    x2 = ofcolor(I, x1)
    O = subgrid(x2, I)
    return O


def solve_f25ffba3(I):
    x1 = bottomhalf(I)
    x2 = hmirror(x1)
    O = vconcat(x2, x1)
    return O


def solve_c1d99e64(I):
    x1 = frontiers(I)
    x2 = merge(x1)
    O = fill(I, 2, x2)
    return O


def solve_b91ae062(I):
    x1 = numcolors(I)
    x2 = decrement(x1)
    O = upscale(I, x2)
    return O


def solve_3aa6fb7a(I):
    x1 = objects(I, T, F, T)
    x2 = mapply(corners, x1)
    O = underfill(I, 1, x2)
    return O


def solve_7b7f7511(I):
    x1 = portrait(I)
    x2 = branch(x1, tophalf, lefthalf)
    O = x2(I)
    return O


def solve_4258a5f9(I):
    x1 = ofcolor(I, 5)
    x2 = mapply(neighbors, x1)
    O = fill(I, 1, x2)
    return O


def solve_2dc579da(I):
    x1 = vsplit(I, 2)
    x2 = rbind(hsplit, 2)
    x3 = mapply(x2, x1)
    O = argmax(x3, numcolors)
    return O


def solve_28bf18c6(I):
    x1 = objects(I, T, T, T)
    x2 = first(x1)
    x3 = subgrid(x2, I)
    O = hconcat(x3, x3)
    return O


def solve_3af2c5a8(I):
    x1 = vmirror(I)
    x2 = hconcat(I, x1)
    x3 = hmirror(x2)
    O = vconcat(x2, x3)
    return O


def solve_44f52bb0(I):
    x1 = vmirror(I)
    x2 = equality(x1, I)
    x3 = branch(x2, 1, 7)
    O = canvas(x3, UNITY)
    return O


def solve_62c24649(I):
    x1 = vmirror(I)
    x2 = hconcat(I, x1)
    x3 = hmirror(x2)
    O = vconcat(x2, x3)
    return O


def solve_67e8384a(I):
    x1 = vmirror(I)
    x2 = hconcat(I, x1)
    x3 = hmirror(x2)
    O = vconcat(x2, x3)
    return O


def solve_7468f01a(I):
    x1 = objects(I, F, T, T)
    x2 = first(x1)
    x3 = subgrid(x2, I)
    O = vmirror(x3)
    return O


def solve_662c240a(I):
    x1 = vsplit(I, 3)
    x2 = fork(equality, dmirror, identity)
    x3 = compose(flip, x2)
    O = extract(x1, x3)
    return O


def solve_42a50994(I):
    x1 = objects(I, T, T, T)
    x2 = sizefilter(x1, 1)
    x3 = merge(x2)
    O = cover(I, x3)
    return O


def solve_56ff96f3(I):
    x1 = fgpartition(I)
    x2 = fork(recolor, color, backdrop)
    x3 = mapply(x2, x1)
    O = paint(I, x3)
    return O


def solve_50cb2852(I):
    x1 = objects(I, T, F, T)
    x2 = compose(backdrop, inbox)
    x3 = mapply(x2, x1)
    O = fill(I, 8, x3)
    return O


def solve_4347f46a(I):
    x1 = objects(I, T, F, T)
    x2 = fork(difference, toindices, box)
    x3 = mapply(x2, x1)
    O = fill(I, 0, x3)
    return O


def solve_46f33fce(I):
    x1 = rot180(I)
    x2 = downscale(x1, 2)
    x3 = rot180(x2)
    O = upscale(x3, 4)
    return O


def solve_a740d043(I):
    x1 = objects(I, T, T, T)
    x2 = merge(x1)
    x3 = subgrid(x2, I)
    O = replace(x3, 1, 0)
    return O


def solve_a79310a0(I):
    x1 = objects(I, T, F, T)
    x2 = first(x1)
    x3 = move(I, x2, DOWN)
    O = replace(x3, 8, 2)
    return O


def solve_aabf363d(I):
    x1 = leastcolor(I)
    x2 = replace(I, x1, 0)
    x3 = leastcolor(x2)
    O = replace(x2, x3, x1)
    return O


def solve_ae4f1146(I):
    x1 = objects(I, F, F, T)
    x2 = rbind(colorcount, 1)
    x3 = argmax(x1, x2)
    O = subgrid(x3, I)
    return O


def solve_b27ca6d3(I):
    x1 = objects(I, T, F, T)
    x2 = sizefilter(x1, 2)
    x3 = mapply(outbox, x2)
    O = fill(I, 3, x3)
    return O


def solve_ce22a75a(I):
    x1 = objects(I, T, F, T)
    x2 = apply(outbox, x1)
    x3 = mapply(backdrop, x2)
    O = fill(I, 1, x3)
    return O


def solve_dc1df850(I):
    x1 = objects(I, T, F, T)
    x2 = colorfilter(x1, 2)
    x3 = mapply(outbox, x2)
    O = fill(I, 1, x3)
    return O


def solve_f25fbde4(I):
    x1 = objects(I, T, T, T)
    x2 = first(x1)
    x3 = subgrid(x2, I)
    O = upscale(x3, 2)
    return O


def solve_44d8ac46(I):
    x1 = objects(I, T, F, T)
    x2 = apply(delta, x1)
    x3 = mfilter(x2, square)
    O = fill(I, 2, x3)
    return O


def solve_1e0a9b12(I):
    x1 = rot270(I)
    x2 = rbind(order, identity)
    x3 = apply(x2, x1)
    O = rot90(x3)
    return O


def solve_0d3d703e(I):
    x1 = switch(I, 3, 4)
    x2 = switch(x1, 8, 9)
    x3 = switch(x2, 2, 6)
    O = switch(x3, 1, 5)
    return O


def solve_3618c87e(I):
    x1 = objects(I, T, F, T)
    x2 = sizefilter(x1, 1)
    x3 = merge(x2)
    O = move(I, x3, G2x0)
    return O


def solve_1c786137(I):
    x1 = objects(I, T, F, F)
    x2 = argmax(x1, height)
    x3 = subgrid(x2, I)
    O = trim(x3)
    return O


def solve_8efcae92(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, 1)
    x3 = compose(size, delta)
    x4 = argmax(x2, x3)
    O = subgrid(x4, I)
    return O


def solve_445eab21(I):
    x1 = objects(I, T, F, T)
    x2 = fork(multiply, height, width)
    x3 = argmax(x1, x2)
    x4 = color(x3)
    O = canvas(x4, G2x2)
    return O


def solve_6f8cd79b(I):
    x1 = asindices(I)
    x2 = apply(initset, x1)
    x3 = rbind(bordering, I)
    x4 = mfilter(x2, x3)
    O = fill(I, 8, x4)
    return O


def solve_2013d3e2(I):
    x1 = objects(I, F, T, T)
    x2 = first(x1)
    x3 = subgrid(x2, I)
    x4 = lefthalf(x3)
    O = tophalf(x4)
    return O


def solve_41e4d17e(I):
    x1 = objects(I, T, F, T)
    x2 = fork(combine, vfrontier, hfrontier)
    x3 = compose(x2, center)
    x4 = mapply(x3, x1)
    O = underfill(I, 6, x4)
    return O


def solve_9565186b(I):
    x1 = shape(I)
    x2 = objects(I, T, F, F)
    x3 = argmax(x2, size)
    x4 = canvas(5, x1)
    O = paint(x4, x3)
    return O


def solve_aedd82e4(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, 2)
    x3 = sizefilter(x2, 1)
    x4 = merge(x3)
    O = fill(I, 1, x4)
    return O


def solve_bb43febb(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, 5)
    x3 = compose(backdrop, inbox)
    x4 = mapply(x3, x2)
    O = fill(I, 2, x4)
    return O


def solve_e98196ab(I):
    x1 = tophalf(I)
    x2 = bottomhalf(I)
    x3 = objects(x1, T, F, T)
    x4 = merge(x3)
    O = paint(x2, x4)
    return O


def solve_f76d97a5(I):
    x1 = palette(I)
    x2 = first(x1)
    x3 = last(x1)
    x4 = switch(I, x2, x3)
    O = replace(x4, 5, 0)
    return O


def solve_ce9e57f2(I):
    x1 = objects(I, T, F, T)
    x2 = fork(connect, ulcorner, centerofmass)
    x3 = mapply(x2, x1)
    x4 = fill(I, 8, x3)
    O = switch(x4, 8, 2)
    return O


def solve_22eb0ac0(I):
    x1 = fgpartition(I)
    x2 = fork(recolor, color, backdrop)
    x3 = apply(x2, x1)
    x4 = mfilter(x3, hline)
    O = paint(I, x4)
    return O


def solve_9f236235(I):
    x1 = compress(I)
    x2 = objects(I, T, F, F)
    x3 = vmirror(x1)
    x4 = valmin(x2, width)
    O = downscale(x3, x4)
    return O


def solve_a699fb00(I):
    x1 = ofcolor(I, 1)
    x2 = shift(x1, RIGHT)
    x3 = shift(x1, LEFT)
    x4 = intersection(x2, x3)
    O = fill(I, 2, x4)
    return O


def solve_46442a0e(I):
    x1 = rot90(I)
    x2 = rot180(I)
    x3 = rot270(I)
    x4 = hconcat(I, x1)
    x5 = hconcat(x3, x2)
    O = vconcat(x4, x5)
    return O


def solve_7fe24cdd(I):
    x1 = rot90(I)
    x2 = rot180(I)
    x3 = rot270(I)
    x4 = hconcat(I, x1)
    x5 = hconcat(x3, x2)
    O = vconcat(x4, x5)
    return O


def solve_0ca9ddb6(I):
    x1 = ofcolor(I, 1)
    x2 = ofcolor(I, 2)
    x3 = mapply(dneighbors, x1)
    x4 = mapply(ineighbors, x2)
    x5 = fill(I, 7, x3)
    O = fill(x5, 4, x4)
    return O


def solve_543a7ed5(I):
    x1 = objects(I, T, F, T)
    x2 = colorfilter(x1, 6)
    x3 = mapply(outbox, x2)
    x4 = fill(I, 3, x3)
    x5 = mapply(delta, x2)
    O = fill(x4, 4, x5)
    return O


def solve_0520fde7(I):
    x1 = vmirror(I)
    x2 = lefthalf(x1)
    x3 = righthalf(x1)
    x4 = vmirror(x3)
    x5 = cellwise(x2, x4, 0)
    O = replace(x5, 1, 2)
    return O


def solve_dae9d2b5(I):
    x1 = lefthalf(I)
    x2 = righthalf(I)
    x3 = ofcolor(x1, 4)
    x4 = ofcolor(x2, 3)
    x5 = combine(x3, x4)
    O = fill(x1, 6, x5)
    return O


def solve_8d5021e8(I):
    x1 = vmirror(I)
    x2 = hconcat(x1, I)
    x3 = hmirror(x2)
    x4 = vconcat(x2, x3)
    x5 = vconcat(x4, x2)
    O = hmirror(x5)
    return O


def solve_928ad970(I):
    x1 = ofcolor(I, 5)
    x2 = subgrid(x1, I)
    x3 = trim(x2)
    x4 = leastcolor(x3)
    x5 = inbox(x1)
    O = fill(I, x4, x5)
    return O


def solve_b60334d2(I):
    x1 = ofcolor(I, 5)
    x2 = replace(I, 5, 0)
    x3 = mapply(dneighbors, x1)
    x4 = mapply(ineighbors, x1)
    x5 = fill(x2, 1, x3)
    O = fill(x5, 5, x4)
    return O


def solve_b94a9452(I):
    x1 = objects(I, F, F, T)
    x2 = first(x1)
    x3 = subgrid(x2, I)
    x4 = leastcolor(x3)
    x5 = mostcolor(x3)
    O = switch(x3, x4, x5)
    return O


def solve_d037b0a7(I):
    x1 = objects(I, T, F, T)
    x2 = rbind(shoot, DOWN)
    x3 = compose(x2, center)
    x4 = fork(recolor, color, x3)
    x5 = mapply(x4, x1)
    O = paint(I, x5)
    return O


def solve_d0f5fe59(I):
    x1 = objects(I, T, F, T)
    x2 = size(x1)
    x3 = astuple(x2, x2)
    x4 = canvas(0, x3)
    x5 = shoot(ORIGIN, UNITY)
    O = fill(x4, 8, x5)
    return O


def solve_e3497940(I):
    x1 = lefthalf(I)
    x2 = righthalf(I)
    x3 = vmirror(x2)
    x4 = objects(x3, T, F, T)
    x5 = merge(x4)
    O = paint(x1, x5)
    return O


def solve_e9afcf9a(I):
    x1 = astuple(2, 1)
    x2 = crop(I, ORIGIN, x1)
    x3 = hmirror(x2)
    x4 = hconcat(x2, x3)
    x5 = hconcat(x4, x4)
    O = hconcat(x5, x4)
    return O


def solve_48d8fb45(I):
    x1 = objects(I, T, T, T)
    x2 = matcher(size, 1)
    x3 = extract(x1, x2)
    x4 = lbind(adjacent, x3)
    x5 = extract(x1, x4)
    O = subgrid(x5, I)
    return O


def solve_d406998b(I):
    x1 = vmirror(I)
    x2 = ofcolor(x1, 5)
    x3 = compose(even, last)
    x4 = sfilter(x2, x3)
    x5 = fill(x1, 3, x4)
    O = vmirror(x5)
    return O


def solve_5117e062(I):
    x1 = objects(I, F, T, T)
    x2 = matcher(numcolors, 2)
    x3 = extract(x1, x2)
    x4 = subgrid(x3, I)
    x5 = mostcolor(x3)
    O = replace(x4, 8, x5)
    return O


def solve_3906de3d(I):
    x1 = rot270(I)
    x2 = rbind(order, identity)
    x3 = switch(x1, 1, 2)
    x4 = apply(x2, x3)
    x5 = switch(x4, 1, 2)
    O = cmirror(x5)
    return O


def solve_00d62c1b(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, 0)
    x3 = rbind(bordering, I)
    x4 = compose(flip, x3)
    x5 = mfilter(x2, x4)
    O = fill(I, 4, x5)
    return O


def solve_7b6016b9(I):
    x1 = objects(I, T, F, F)
    x2 = rbind(bordering, I)
    x3 = compose(flip, x2)
    x4 = mfilter(x1, x3)
    x5 = fill(I, 2, x4)
    O = replace(x5, 0, 3)
    return O


def solve_67385a82(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, 3)
    x3 = sizefilter(x2, 1)
    x4 = difference(x2, x3)
    x5 = merge(x4)
    O = fill(I, 8, x5)
    return O


def solve_a5313dff(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, 0)
    x3 = rbind(bordering, I)
    x4 = compose(flip, x3)
    x5 = mfilter(x2, x4)
    O = fill(I, 1, x5)
    return O


def solve_ea32f347(I):
    x1 = objects(I, T, F, T)
    x2 = replace(I, 5, 4)
    x3 = argmin(x1, size)
    x4 = argmax(x1, size)
    x5 = fill(x2, 1, x4)
    O = fill(x5, 2, x3)
    return O


def solve_d631b094(I):
    x1 = palette(I)
    x2 = other(x1, 0)
    x3 = ofcolor(I, x2)
    x4 = size(x3)
    x5 = astuple(1, x4)
    O = canvas(x2, x5)
    return O


def solve_10fcaaa3(I):
    x1 = leastcolor(I)
    x2 = hconcat(I, I)
    x3 = vconcat(x2, x2)
    x4 = ofcolor(x3, x1)
    x5 = mapply(ineighbors, x4)
    O = underfill(x3, 8, x5)
    return O


def solve_007bbfb7(I):
    x1 = hupscale(I, 3)
    x2 = vupscale(x1, 3)
    x3 = hconcat(I, I)
    x4 = hconcat(x3, I)
    x5 = vconcat(x4, x4)
    x6 = vconcat(x5, x4)
    O = cellwise(x2, x6, 0)
    return O


def solve_496994bd(I):
    x1 = width(I)
    x2 = height(I)
    x3 = halve(x2)
    x4 = astuple(x3, x1)
    x5 = crop(I, ORIGIN, x4)
    x6 = hmirror(x5)
    O = vconcat(x5, x6)
    return O


def solve_1f876c06(I):
    x1 = fgpartition(I)
    x2 = compose(last, first)
    x3 = power(last, 2)
    x4 = fork(connect, x2, x3)
    x5 = fork(recolor, color, x4)
    x6 = mapply(x5, x1)
    O = paint(I, x6)
    return O


def solve_05f2a901(I):
    x1 = objects(I, T, F, T)
    x2 = colorfilter(x1, 2)
    x3 = first(x2)
    x4 = colorfilter(x1, 8)
    x5 = first(x4)
    x6 = gravitate(x3, x5)
    O = move(I, x3, x6)
    return O


def solve_39a8645d(I):
    x1 = objects(I, T, T, T)
    x2 = totuple(x1)
    x3 = apply(color, x2)
    x4 = mostcommon(x3)
    x5 = matcher(color, x4)
    x6 = extract(x1, x5)
    O = subgrid(x6, I)
    return O


def solve_1b2d62fb(I):
    x1 = lefthalf(I)
    x2 = righthalf(I)
    x3 = ofcolor(x1, 0)
    x4 = ofcolor(x2, 0)
    x5 = intersection(x3, x4)
    x6 = replace(x1, 9, 0)
    O = fill(x6, 8, x5)
    return O


def solve_90c28cc7(I):
    x1 = objects(I, F, F, T)
    x2 = first(x1)
    x3 = subgrid(x2, I)
    x4 = dedupe(x3)
    x5 = rot90(x4)
    x6 = dedupe(x5)
    O = rot270(x6)
    return O


def solve_b6afb2da(I):
    x1 = objects(I, T, F, F)
    x2 = replace(I, 5, 2)
    x3 = colorfilter(x1, 5)
    x4 = mapply(box, x3)
    x5 = fill(x2, 4, x4)
    x6 = mapply(corners, x3)
    O = fill(x5, 1, x6)
    return O


def solve_b9b7f026(I):
    x1 = objects(I, T, F, F)
    x2 = argmin(x1, size)
    x3 = rbind(adjacent, x2)
    x4 = remove(x2, x1)
    x5 = extract(x4, x3)
    x6 = color(x5)
    O = canvas(x6, UNITY)
    return O


def solve_ba97ae07(I):
    x1 = objects(I, T, F, T)
    x2 = totuple(x1)
    x3 = apply(color, x2)
    x4 = mostcommon(x3)
    x5 = ofcolor(I, x4)
    x6 = backdrop(x5)
    O = fill(I, x4, x6)
    return O


def solve_c9f8e694(I):
    x1 = height(I)
    x2 = width(I)
    x3 = ofcolor(I, 0)
    x4 = astuple(x1, 1)
    x5 = crop(I, ORIGIN, x4)
    x6 = hupscale(x5, x2)
    O = fill(x6, 0, x3)
    return O


def solve_d23f8c26(I):
    x1 = asindices(I)
    x2 = width(I)
    x3 = halve(x2)
    x4 = matcher(last, x3)
    x5 = compose(flip, x4)
    x6 = sfilter(x1, x5)
    O = fill(I, 0, x6)
    return O


def solve_d5d6de2d(I):
    x1 = objects(I, T, F, T)
    x2 = sfilter(x1, square)
    x3 = difference(x1, x2)
    x4 = compose(backdrop, inbox)
    x5 = mapply(x4, x3)
    x6 = replace(I, 2, 0)
    O = fill(x6, 3, x5)
    return O


def solve_dbc1a6ce(I):
    x1 = ofcolor(I, 1)
    x2 = product(x1, x1)
    x3 = fork(connect, first, last)
    x4 = apply(x3, x2)
    x5 = fork(either, vline, hline)
    x6 = mfilter(x4, x5)
    O = underfill(I, 8, x6)
    return O


def solve_ded97339(I):
    x1 = ofcolor(I, 8)
    x2 = product(x1, x1)
    x3 = fork(connect, first, last)
    x4 = apply(x3, x2)
    x5 = fork(either, vline, hline)
    x6 = mfilter(x4, x5)
    O = underfill(I, 8, x6)
    return O


def solve_ea786f4a(I):
    x1 = width(I)
    x2 = shoot(ORIGIN, UNITY)
    x3 = decrement(x1)
    x4 = tojvec(x3)
    x5 = shoot(x4, DOWN_LEFT)
    x6 = combine(x2, x5)
    O = fill(I, 0, x6)
    return O


def solve_08ed6ac7(I):
    x1 = objects(I, T, F, T)
    x2 = totuple(x1)
    x3 = order(x1, height)
    x4 = size(x2)
    x5 = interval(x4, 0,  -1)
    x6 = mpapply(recolor, x5, x3)
    O = paint(I, x6)
    return O


def solve_40853293(I):
    x1 = partition(I)
    x2 = fork(recolor, color, backdrop)
    x3 = apply(x2, x1)
    x4 = mfilter(x3, hline)
    x5 = mfilter(x3, vline)
    x6 = paint(I, x4)
    O = paint(x6, x5)
    return O


def solve_5521c0d9(I):
    x1 = objects(I, T, F, T)
    x2 = merge(x1)
    x3 = cover(I, x2)
    x4 = chain(toivec, invert, height)
    x5 = fork(shift, identity, x4)
    x6 = mapply(x5, x1)
    O = paint(x3, x6)
    return O


def solve_f8ff0b80(I):
    x1 = objects(I, T, T, T)
    x2 = order(x1, size)
    x3 = apply(color, x2)
    x4 = rbind(canvas, UNITY)
    x5 = apply(x4, x3)
    x6 = merge(x5)
    O = hmirror(x6)
    return O


def solve_85c4e7cd(I):
    x1 = objects(I, T, F, F)
    x2 = compose(invert, size)
    x3 = order(x1, size)
    x4 = order(x1, x2)
    x5 = apply(color, x4)
    x6 = mpapply(recolor, x5, x3)
    O = paint(I, x6)
    return O


def solve_d2abd087(I):
    x1 = objects(I, T, F, T)
    x2 = matcher(size, 6)
    x3 = compose(flip, x2)
    x4 = mfilter(x1, x2)
    x5 = mfilter(x1, x3)
    x6 = fill(I, 2, x4)
    O = fill(x6, 1, x5)
    return O


def solve_017c7c7b(I):
    x1 = tophalf(I)
    x2 = bottomhalf(I)
    x3 = equality(x1, x2)
    x4 = crop(I, G2x0, G3x3)
    x5 = branch(x3, x2, x4)
    x6 = vconcat(I, x5)
    O = replace(x6, 1, 2)
    return O


def solve_363442ee(I):
    x1 = ofcolor(I, 1)
    x2 = crop(I, ORIGIN, G3x3)
    x3 = asobject(x2)
    x4 = lbind(shift, x3)
    x5 = compose(x4, decrement)
    x6 = mapply(x5, x1)
    O = paint(I, x6)
    return O


def solve_5168d44c(I):
    x1 = ofcolor(I, 3)
    x2 = height(x1)
    x3 = equality(x2, 1)
    x4 = branch(x3, G0x2, G2x0)
    x5 = ofcolor(I, 2)
    x6 = recolor(2, x5)
    O = move(I, x6, x4)
    return O


def solve_e9614598(I):
    x1 = ofcolor(I, 1)
    x2 = fork(add, first, last)
    x3 = x2(x1)
    x4 = halve(x3)
    x5 = dneighbors(x4)
    x6 = insert(x4, x5)
    O = fill(I, 3, x6)
    return O


def solve_d9fac9be(I):
    x1 = palette(I)
    x2 = objects(I, T, F, T)
    x3 = argmax(x2, size)
    x4 = color(x3)
    x5 = remove(0, x1)
    x6 = other(x5, x4)
    O = canvas(x6, UNITY)
    return O


def solve_e50d258f(I):
    x1 = width(I)
    x2 = astuple(9, x1)
    x3 = canvas(0, x2)
    x4 = vconcat(I, x3)
    x5 = objects(x4, F, F, T)
    x6 = rbind(colorcount, 2)
    x7 = argmax(x5, x6)
    O = subgrid(x7, I)
    return O


def solve_810b9b61(I):
    x1 = objects(I, T, T, T)
    x2 = apply(toindices, x1)
    x3 = fork(either, vline, hline)
    x4 = sfilter(x2, x3)
    x5 = difference(x2, x4)
    x6 = fork(equality, identity, box)
    x7 = mfilter(x5, x6)
    O = fill(I, 3, x7)
    return O


def solve_54d82841(I):
    x1 = height(I)
    x2 = objects(I, T, F, T)
    x3 = compose(last, center)
    x4 = apply(x3, x2)
    x5 = decrement(x1)
    x6 = lbind(astuple, x5)
    x7 = apply(x6, x4)
    O = fill(I, 4, x7)
    return O


def solve_60b61512(I):
    x1 = objects(I, T, T, T)
    x2 = mapply(delta, x1)
    O = fill(I, 7, x2)
    return O


def solve_25d8a9c8(I):
    x1 = asindices(I)
    x2 = objects(I, T, F, F)
    x3 = sizefilter(x2, 3)
    x4 = mfilter(x3, hline)
    x5 = toindices(x4)
    x6 = difference(x1, x5)
    x7 = fill(I, 5, x5)
    O = fill(x7, 0, x6)
    return O


def solve_239be575(I):
    x1 = objects(I, F, T, T)
    x2 = lbind(contained, 2)
    x3 = compose(x2, palette)
    x4 = sfilter(x1, x3)
    x5 = size(x4)
    x6 = greater(x5, 1)
    x7 = branch(x6, 0, 8)
    O = canvas(x7, UNITY)
    return O


def solve_67a423a3(I):
    x1 = leastcolor(I)
    x2 = objects(I, T, F, T)
    x3 = colorfilter(x2, x1)
    x4 = merge(x3)
    x5 = delta(x4)
    x6 = first(x5)
    x7 = neighbors(x6)
    O = fill(I, 4, x7)
    return O


def solve_5c0a986e(I):
    x1 = ofcolor(I, 2)
    x2 = ofcolor(I, 1)
    x3 = lrcorner(x1)
    x4 = ulcorner(x2)
    x5 = shoot(x3, UNITY)
    x6 = shoot(x4, NEG_UNITY)
    x7 = fill(I, 2, x5)
    O = fill(x7, 1, x6)
    return O


def solve_6430c8c4(I):
    x1 = tophalf(I)
    x2 = bottomhalf(I)
    x3 = astuple(4, 4)
    x4 = ofcolor(x1, 0)
    x5 = ofcolor(x2, 0)
    x6 = intersection(x4, x5)
    x7 = canvas(0, x3)
    O = fill(x7, 3, x6)
    return O


def solve_94f9d214(I):
    x1 = tophalf(I)
    x2 = bottomhalf(I)
    x3 = ofcolor(x1, 0)
    x4 = ofcolor(x2, 0)
    x5 = astuple(4, 4)
    x6 = canvas(0, x5)
    x7 = intersection(x3, x4)
    O = fill(x6, 2, x7)
    return O


def solve_a1570a43(I):
    x1 = ofcolor(I, 2)
    x2 = ofcolor(I, 3)
    x3 = recolor(2, x1)
    x4 = ulcorner(x2)
    x5 = ulcorner(x1)
    x6 = subtract(x4, x5)
    x7 = increment(x6)
    O = move(I, x3, x7)
    return O


def solve_ce4f8723(I):
    x1 = tophalf(I)
    x2 = bottomhalf(I)
    x3 = ofcolor(x1, 0)
    x4 = ofcolor(x2, 0)
    x5 = intersection(x3, x4)
    x6 = astuple(4, 4)
    x7 = canvas(3, x6)
    O = fill(x7, 0, x5)
    return O


def solve_d13f3404(I):
    x1 = objects(I, T, F, T)
    x2 = rbind(shoot, UNITY)
    x3 = compose(x2, center)
    x4 = fork(recolor, color, x3)
    x5 = mapply(x4, x1)
    x6 = astuple(6, 6)
    x7 = canvas(0, x6)
    O = paint(x7, x5)
    return O


def solve_dc433765(I):
    x1 = ofcolor(I, 3)
    x2 = ofcolor(I, 4)
    x3 = first(x1)
    x4 = first(x2)
    x5 = subtract(x4, x3)
    x6 = sign(x5)
    x7 = recolor(3, x1)
    O = move(I, x7, x6)
    return O


def solve_f2829549(I):
    x1 = lefthalf(I)
    x2 = righthalf(I)
    x3 = ofcolor(x1, 0)
    x4 = ofcolor(x2, 0)
    x5 = intersection(x3, x4)
    x6 = shape(x1)
    x7 = canvas(0, x6)
    O = fill(x7, 3, x5)
    return O


def solve_fafffa47(I):
    x1 = tophalf(I)
    x2 = bottomhalf(I)
    x3 = shape(x2)
    x4 = ofcolor(x1, 0)
    x5 = ofcolor(x2, 0)
    x6 = intersection(x4, x5)
    x7 = canvas(0, x3)
    O = fill(x7, 2, x6)
    return O


def solve_fcb5c309(I):
    x1 = leastcolor(I)
    x2 = objects(I, T, F, T)
    x3 = colorfilter(x2, x1)
    x4 = difference(x2, x3)
    x5 = argmax(x4, size)
    x6 = color(x5)
    x7 = subgrid(x5, I)
    O = replace(x7, x6, x1)
    return O


def solve_ff805c23(I):
    x1 = hmirror(I)
    x2 = vmirror(I)
    x3 = ofcolor(I, 1)
    x4 = subgrid(x3, x1)
    x5 = subgrid(x3, x2)
    x6 = palette(x4)
    x7 = contained(1, x6)
    O = branch(x7, x5, x4)
    return O


def solve_e76a88a6(I):
    x1 = objects(I, F, F, T)
    x2 = argmax(x1, numcolors)
    x3 = normalize(x2)
    x4 = remove(x2, x1)
    x5 = apply(ulcorner, x4)
    x6 = lbind(shift, x3)
    x7 = mapply(x6, x5)
    O = paint(I, x7)
    return O


def solve_7c008303(I):
    x1 = ofcolor(I, 3)
    x2 = subgrid(x1, I)
    x3 = ofcolor(x2, 0)
    x4 = replace(I, 3, 0)
    x5 = replace(x4, 8, 0)
    x6 = compress(x5)
    x7 = upscale(x6, 3)
    O = fill(x7, 0, x3)
    return O


def solve_7f4411dc(I):
    x1 = leastcolor(I)
    x2 = ofcolor(I, x1)
    x3 = rbind(difference, x2)
    x4 = rbind(greater, 2)
    x5 = chain(x4, size, x3)
    x6 = compose(x5, dneighbors)
    x7 = sfilter(x2, x6)
    O = fill(I, 0, x7)
    return O


def solve_b230c067(I):
    x1 = objects(I, T, T, T)
    x2 = totuple(x1)
    x3 = apply(normalize, x2)
    x4 = leastcommon(x3)
    x5 = matcher(normalize, x4)
    x6 = extract(x1, x5)
    x7 = replace(I, 8, 1)
    O = fill(x7, 2, x6)
    return O


def solve_e8593010(I):
    x1 = objects(I, T, F, T)
    x2 = sizefilter(x1, 1)
    x3 = sizefilter(x1, 2)
    x4 = merge(x2)
    x5 = fill(I, 3, x4)
    x6 = merge(x3)
    x7 = fill(x5, 2, x6)
    O = replace(x7, 0, 1)
    return O


def solve_6d75e8bb(I):
    x1 = objects(I, T, F, T)
    x2 = first(x1)
    x3 = ulcorner(x2)
    x4 = subgrid(x2, I)
    x5 = replace(x4, 0, 2)
    x6 = asobject(x5)
    x7 = shift(x6, x3)
    O = paint(I, x7)
    return O


def solve_3f7978a0(I):
    x1 = fgpartition(I)
    x2 = matcher(color, 5)
    x3 = extract(x1, x2)
    x4 = ulcorner(x3)
    x5 = subtract(x4, DOWN)
    x6 = shape(x3)
    x7 = add(x6, G2x0)
    O = crop(I, x5, x7)
    return O


def solve_1190e5a7(I):
    x1 = mostcolor(I)
    x2 = frontiers(I)
    x3 = sfilter(x2, vline)
    x4 = difference(x2, x3)
    x5 = astuple(x4, x3)
    x6 = apply(size, x5)
    x7 = increment(x6)
    O = canvas(x1, x7)
    return O


def solve_6e02f1e3(I):
    x1 = numcolors(I)
    x2 = canvas(0, G3x3)
    x3 = equality(x1, 3)
    x4 = equality(x1, 2)
    x5 = branch(x3, G2x0, ORIGIN)
    x6 = branch(x4, G2x2, G0x2)
    x7 = connect(x5, x6)
    O = fill(x2, 5, x7)
    return O


def solve_a61f2674(I):
    x1 = objects(I, T, F, T)
    x2 = argmax(x1, size)
    x3 = argmin(x1, size)
    x4 = replace(I, 5, 0)
    x5 = recolor(1, x2)
    x6 = recolor(2, x3)
    x7 = combine(x5, x6)
    O = paint(x4, x7)
    return O


def solve_fcc82909(I):
    x1 = objects(I, F, T, T)
    x2 = rbind(add, DOWN)
    x3 = compose(x2, llcorner)
    x4 = compose(toivec, numcolors)
    x5 = fork(add, lrcorner, x4)
    x6 = fork(astuple, x3, x5)
    x7 = compose(box, x6)
    x8 = mapply(x7, x1)
    O = fill(I, 3, x8)
    return O


def solve_72ca375d(I):
    x1 = objects(I, T, T, T)
    x2 = totuple(x1)
    x3 = rbind(subgrid, I)
    x4 = apply(x3, x2)
    x5 = apply(vmirror, x4)
    x6 = papply(equality, x4, x5)
    x7 = pair(x4, x6)
    x8 = extract(x7, last)
    O = first(x8)
    return O


def solve_253bf280(I):
    x1 = ofcolor(I, 8)
    x2 = prapply(connect, x1, x1)
    x3 = rbind(greater, 1)
    x4 = compose(x3, size)
    x5 = sfilter(x2, x4)
    x6 = fork(either, vline, hline)
    x7 = mfilter(x5, x6)
    x8 = fill(I, 3, x7)
    O = fill(x8, 8, x1)
    return O


def solve_694f12f3(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, 4)
    x3 = compose(backdrop, inbox)
    x4 = argmin(x2, size)
    x5 = argmax(x2, size)
    x6 = x3(x4)
    x7 = x3(x5)
    x8 = fill(I, 1, x6)
    O = fill(x8, 2, x7)
    return O


def solve_1f642eb9(I):
    x1 = objects(I, T, F, T)
    x2 = sizefilter(x1, 1)
    x3 = difference(x1, x2)
    x4 = first(x3)
    x5 = rbind(gravitate, x4)
    x6 = compose(crement, x5)
    x7 = fork(shift, identity, x6)
    x8 = mapply(x7, x2)
    O = paint(I, x8)
    return O


def solve_31aa019c(I):
    x1 = leastcolor(I)
    x2 = ofcolor(I, x1)
    x3 = first(x2)
    x4 = neighbors(x3)
    x5 = astuple(10, 10)
    x6 = canvas(0, x5)
    x7 = initset(x3)
    x8 = fill(x6, x1, x7)
    O = fill(x8, 2, x4)
    return O


def solve_27a28665(I):
    x1 = objects(I, T, F, F)
    x2 = valmax(x1, size)
    x3 = equality(x2, 1)
    x4 = equality(x2, 4)
    x5 = equality(x2, 5)
    x6 = branch(x3, 2, 1)
    x7 = branch(x4, 3, x6)
    x8 = branch(x5, 6, x7)
    O = canvas(x8, UNITY)
    return O


def solve_7ddcd7ec(I):
    x1 = objects(I, T, F, T)
    x2 = sizefilter(x1, 1)
    x3 = difference(x1, x2)
    x4 = first(x3)
    x5 = color(x4)
    x6 = lbind(position, x4)
    x7 = fork(shoot, center, x6)
    x8 = mapply(x7, x2)
    O = fill(I, x5, x8)
    return O


def solve_3bd67248(I):
    x1 = height(I)
    x2 = decrement(x1)
    x3 = decrement(x2)
    x4 = astuple(x3, 1)
    x5 = astuple(x2, 1)
    x6 = shoot(x4, UP_RIGHT)
    x7 = shoot(x5, RIGHT)
    x8 = fill(I, 2, x6)
    O = fill(x8, 4, x7)
    return O


def solve_73251a56(I):
    x1 = dmirror(I)
    x2 = papply(pair, I, x1)
    x3 = lbind(apply, maximum)
    x4 = apply(x3, x2)
    x5 = mostcolor(x4)
    x6 = replace(x4, 0, x5)
    x7 = index(x6, ORIGIN)
    x8 = shoot(ORIGIN, UNITY)
    O = fill(x6, x7, x8)
    return O


def solve_25d487eb(I):
    x1 = leastcolor(I)
    x2 = objects(I, T, F, T)
    x3 = ofcolor(I, x1)
    x4 = center(x3)
    x5 = merge(x2)
    x6 = center(x5)
    x7 = subtract(x6, x4)
    x8 = shoot(x4, x7)
    O = underfill(I, x1, x8)
    return O


def solve_8f2ea7aa(I):
    x1 = objects(I, T, F, T)
    x2 = merge(x1)
    x3 = subgrid(x2, I)
    x4 = upscale(x3, 3)
    x5 = hconcat(x3, x3)
    x6 = hconcat(x5, x3)
    x7 = vconcat(x6, x6)
    x8 = vconcat(x7, x6)
    O = cellwise(x4, x8, 0)
    return O


def solve_b8825c91(I):
    x1 = replace(I, 4, 0)
    x2 = dmirror(x1)
    x3 = papply(pair, x1, x2)
    x4 = lbind(apply, maximum)
    x5 = apply(x4, x3)
    x6 = cmirror(x5)
    x7 = papply(pair, x5, x6)
    x8 = apply(x4, x7)
    O = cmirror(x8)
    return O


def solve_cce03e0d(I):
    x1 = upscale(I, 3)
    x2 = hconcat(I, I)
    x3 = hconcat(x2, I)
    x4 = vconcat(x3, x3)
    x5 = vconcat(x4, x3)
    x6 = ofcolor(x1, 0)
    x7 = ofcolor(x1, 1)
    x8 = combine(x6, x7)
    O = fill(x5, 0, x8)
    return O


def solve_d364b489(I):
    x1 = ofcolor(I, 1)
    x2 = shift(x1, DOWN)
    x3 = fill(I, 8, x2)
    x4 = shift(x1, UP)
    x5 = fill(x3, 2, x4)
    x6 = shift(x1, RIGHT)
    x7 = fill(x5, 6, x6)
    x8 = shift(x1, LEFT)
    O = fill(x7, 7, x8)
    return O


def solve_a5f85a15(I):
    x1 = objects(I, T, T, T)
    x2 = interval(1, 9, 1)
    x3 = apply(double, x2)
    x4 = apply(decrement, x3)
    x5 = papply(astuple, x4, x4)
    x6 = apply(ulcorner, x1)
    x7 = lbind(shift, x5)
    x8 = mapply(x7, x6)
    O = fill(I, 4, x8)
    return O


def solve_3ac3eb23(I):
    x1 = objects(I, T, F, T)
    x2 = chain(ineighbors, last, first)
    x3 = fork(recolor, color, x2)
    x4 = mapply(x3, x1)
    x5 = paint(I, x4)
    x6 = vsplit(x5, 3)
    x7 = first(x6)
    x8 = vconcat(x7, x7)
    O = vconcat(x7, x8)
    return O


def solve_444801d8(I):
    x1 = objects(I, T, F, T)
    x2 = colorfilter(x1, 1)
    x3 = rbind(toobject, I)
    x4 = chain(leastcolor, x3, delta)
    x5 = rbind(shift, UP)
    x6 = compose(x5, backdrop)
    x7 = fork(recolor, x4, x6)
    x8 = mapply(x7, x2)
    O = underpaint(I, x8)
    return O


def solve_22168020(I):
    x1 = palette(I)
    x2 = remove(0, x1)
    x3 = lbind(ofcolor, I)
    x4 = lbind(prapply, connect)
    x5 = fork(x4, x3, x3)
    x6 = compose(merge, x5)
    x7 = fork(recolor, identity, x6)
    x8 = mapply(x7, x2)
    O = paint(I, x8)
    return O


def solve_6e82a1ae(I):
    x1 = objects(I, T, F, T)
    x2 = lbind(sizefilter, x1)
    x3 = compose(merge, x2)
    x4 = x3(2)
    x5 = x3(3)
    x6 = x3(4)
    x7 = fill(I, 3, x4)
    x8 = fill(x7, 2, x5)
    O = fill(x8, 1, x6)
    return O


def solve_b2862040(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, 9)
    x3 = colorfilter(x1, 1)
    x4 = rbind(bordering, I)
    x5 = compose(flip, x4)
    x6 = mfilter(x2, x5)
    x7 = rbind(adjacent, x6)
    x8 = mfilter(x3, x7)
    O = fill(I, 8, x8)
    return O


def solve_868de0fa(I):
    x1 = objects(I, T, F, F)
    x2 = sfilter(x1, square)
    x3 = compose(even, height)
    x4 = sfilter(x2, x3)
    x5 = difference(x2, x4)
    x6 = merge(x4)
    x7 = merge(x5)
    x8 = fill(I, 2, x6)
    O = fill(x8, 7, x7)
    return O


def solve_681b3aeb(I):
    x1 = rot270(I)
    x2 = objects(x1, T, F, T)
    x3 = argmax(x2, size)
    x4 = argmin(x2, size)
    x5 = color(x4)
    x6 = canvas(x5, G3x3)
    x7 = normalize(x3)
    x8 = paint(x6, x7)
    O = rot90(x8)
    return O


def solve_8e5a5113(I):
    x1 = crop(I, ORIGIN, G3x3)
    x2 = rot90(x1)
    x3 = rot180(x1)
    x4 = astuple(x2, x3)
    x5 = astuple(4, 8)
    x6 = apply(tojvec, x5)
    x7 = apply(asobject, x4)
    x8 = mpapply(shift, x7, x6)
    O = paint(I, x8)
    return O


def solve_025d127b(I):
    x1 = objects(I, T, F, T)
    x2 = apply(color, x1)
    x3 = merge(x1)
    x4 = lbind(colorfilter, x1)
    x5 = rbind(argmax, rightmost)
    x6 = compose(x5, x4)
    x7 = mapply(x6, x2)
    x8 = difference(x3, x7)
    O = move(I, x8, RIGHT)
    return O


def solve_2281f1f4(I):
    x1 = ofcolor(I, 5)
    x2 = product(x1, x1)
    x3 = power(first, 2)
    x4 = power(last, 2)
    x5 = fork(astuple, x3, x4)
    x6 = apply(x5, x2)
    x7 = urcorner(x1)
    x8 = remove(x7, x6)
    O = underfill(I, 2, x8)
    return O


def solve_cf98881b(I):
    x1 = hsplit(I, 3)
    x2 = first(x1)
    x3 = remove(x2, x1)
    x4 = first(x3)
    x5 = last(x3)
    x6 = ofcolor(x4, 9)
    x7 = ofcolor(x2, 4)
    x8 = fill(x5, 9, x6)
    O = fill(x8, 4, x7)
    return O


def solve_d4f3cd78(I):
    x1 = ofcolor(I, 5)
    x2 = delta(x1)
    x3 = fill(I, 8, x2)
    x4 = box(x1)
    x5 = difference(x4, x1)
    x6 = position(x4, x5)
    x7 = first(x5)
    x8 = shoot(x7, x6)
    O = fill(x3, 8, x8)
    return O


def solve_bda2d7a6(I):
    x1 = partition(I)
    x2 = order(x1, size)
    x3 = apply(color, x2)
    x4 = last(x2)
    x5 = remove(x4, x2)
    x6 = repeat(x4, 1)
    x7 = combine(x6, x5)
    x8 = mpapply(recolor, x3, x7)
    O = paint(I, x8)
    return O


def solve_137eaa0f(I):
    x1 = objects(I, F, T, T)
    x2 = matcher(first, 5)
    x3 = rbind(sfilter, x2)
    x4 = chain(invert, center, x3)
    x5 = fork(shift, identity, x4)
    x6 = canvas(0, G3x3)
    x7 = mapply(x5, x1)
    x8 = shift(x7, UNITY)
    O = paint(x6, x8)
    return O


def solve_6455b5f5(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, 0)
    x3 = argmax(x1, size)
    x4 = valmin(x1, size)
    x5 = sizefilter(x2, x4)
    x6 = recolor(1, x3)
    x7 = merge(x5)
    x8 = paint(I, x6)
    O = fill(x8, 8, x7)
    return O


def solve_b8cdaf2b(I):
    x1 = leastcolor(I)
    x2 = ofcolor(I, x1)
    x3 = shift(x2, UP)
    x4 = ulcorner(x3)
    x5 = urcorner(x3)
    x6 = shoot(x4, NEG_UNITY)
    x7 = shoot(x5, UP_RIGHT)
    x8 = combine(x6, x7)
    O = underfill(I, x1, x8)
    return O


def solve_bd4472b8(I):
    x1 = width(I)
    x2 = astuple(2, x1)
    x3 = crop(I, ORIGIN, x2)
    x4 = tophalf(x3)
    x5 = dmirror(x4)
    x6 = hupscale(x5, x1)
    x7 = repeat(x6, 2)
    x8 = merge(x7)
    O = vconcat(x3, x8)
    return O


def solve_4be741c5(I):
    x1 = portrait(I)
    x2 = branch(x1, dmirror, identity)
    x3 = branch(x1, height, width)
    x4 = x3(I)
    x5 = astuple(1, x4)
    x6 = x2(I)
    x7 = crop(x6, ORIGIN, x5)
    x8 = apply(dedupe, x7)
    O = x2(x8)
    return O


def solve_bbc9ae5d(I):
    x1 = width(I)
    x2 = palette(I)
    x3 = halve(x1)
    x4 = vupscale(I, x3)
    x5 = rbind(shoot, UNITY)
    x6 = other(x2, 0)
    x7 = ofcolor(x4, x6)
    x8 = mapply(x5, x7)
    O = fill(x4, x6, x8)
    return O


def solve_d90796e8(I):
    x1 = objects(I, F, F, T)
    x2 = sizefilter(x1, 2)
    x3 = lbind(contained, 2)
    x4 = compose(x3, palette)
    x5 = mfilter(x2, x4)
    x6 = cover(I, x5)
    x7 = matcher(first, 3)
    x8 = sfilter(x5, x7)
    O = fill(x6, 8, x8)
    return O


def solve_2c608aff(I):
    x1 = leastcolor(I)
    x2 = objects(I, T, F, T)
    x3 = argmax(x2, size)
    x4 = toindices(x3)
    x5 = ofcolor(I, x1)
    x6 = prapply(connect, x4, x5)
    x7 = fork(either, vline, hline)
    x8 = mfilter(x6, x7)
    O = underfill(I, x1, x8)
    return O


def solve_f8b3ba0a(I):
    x1 = compress(I)
    x2 = astuple(3, 1)
    x3 = palette(x1)
    x4 = lbind(colorcount, x1)
    x5 = compose(invert, x4)
    x6 = order(x3, x5)
    x7 = rbind(canvas, UNITY)
    x8 = apply(x7, x6)
    x9 = merge(x8)
    O = crop(x9, DOWN, x2)
    return O


def solve_80af3007(I):
    x1 = objects(I, T, T, T)
    x2 = first(x1)
    x3 = subgrid(x2, I)
    x4 = upscale(x3, 3)
    x5 = hconcat(x3, x3)
    x6 = hconcat(x5, x3)
    x7 = vconcat(x6, x6)
    x8 = vconcat(x7, x6)
    x9 = cellwise(x4, x8, 0)
    O = downscale(x9, 3)
    return O


def solve_83302e8f(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, 0)
    x3 = sfilter(x2, square)
    x4 = difference(x2, x3)
    x5 = merge(x3)
    x6 = recolor(3, x5)
    x7 = merge(x4)
    x8 = recolor(4, x7)
    x9 = paint(I, x6)
    O = paint(x9, x8)
    return O


def solve_1fad071e(I):
    x1 = objects(I, T, F, T)
    x2 = colorfilter(x1, 1)
    x3 = sizefilter(x2, 4)
    x4 = size(x3)
    x5 = subtract(5, x4)
    x6 = astuple(1, x4)
    x7 = canvas(1, x6)
    x8 = astuple(1, x5)
    x9 = canvas(0, x8)
    O = hconcat(x7, x9)
    return O


def solve_11852cab(I):
    x1 = objects(I, T, T, T)
    x2 = merge(x1)
    x3 = hmirror(x2)
    x4 = vmirror(x2)
    x5 = dmirror(x2)
    x6 = cmirror(x2)
    x7 = paint(I, x3)
    x8 = paint(x7, x4)
    x9 = paint(x8, x5)
    O = paint(x9, x6)
    return O


def solve_3428a4f5(I):
    x1 = tophalf(I)
    x2 = bottomhalf(I)
    x3 = astuple(6, 5)
    x4 = ofcolor(x1, 2)
    x5 = ofcolor(x2, 2)
    x6 = combine(x4, x5)
    x7 = intersection(x4, x5)
    x8 = difference(x6, x7)
    x9 = canvas(0, x3)
    O = fill(x9, 3, x8)
    return O


def solve_178fcbfb(I):
    x1 = objects(I, T, F, T)
    x2 = ofcolor(I, 2)
    x3 = mapply(vfrontier, x2)
    x4 = fill(I, 2, x3)
    x5 = colorfilter(x1, 2)
    x6 = difference(x1, x5)
    x7 = compose(hfrontier, center)
    x8 = fork(recolor, color, x7)
    x9 = mapply(x8, x6)
    O = paint(x4, x9)
    return O


def solve_3de23699(I):
    x1 = fgpartition(I)
    x2 = sizefilter(x1, 4)
    x3 = first(x2)
    x4 = difference(x1, x2)
    x5 = first(x4)
    x6 = color(x3)
    x7 = color(x5)
    x8 = subgrid(x3, I)
    x9 = trim(x8)
    O = replace(x9, x7, x6)
    return O


def solve_54d9e175(I):
    x1 = objects(I, T, F, T)
    x2 = sizefilter(x1, 1)
    x3 = compose(neighbors, center)
    x4 = fork(recolor, color, x3)
    x5 = mapply(x4, x2)
    x6 = paint(I, x5)
    x7 = replace(x6, 1, 6)
    x8 = replace(x7, 2, 7)
    x9 = replace(x8, 3, 8)
    O = replace(x9, 4, 9)
    return O


def solve_5ad4f10b(I):
    x1 = objects(I, T, T, T)
    x2 = argmax(x1, size)
    x3 = color(x2)
    x4 = subgrid(x2, I)
    x5 = leastcolor(x4)
    x6 = replace(x4, x5, 0)
    x7 = replace(x6, x3, x5)
    x8 = height(x7)
    x9 = divide(x8, 3)
    O = downscale(x7, x9)
    return O


def solve_623ea044(I):
    x1 = objects(I, T, F, T)
    x2 = first(x1)
    x3 = center(x2)
    x4 = color(x2)
    x5 = astuple(UNITY, NEG_UNITY)
    x6 = astuple(UP_RIGHT, DOWN_LEFT)
    x7 = combine(x5, x6)
    x8 = lbind(shoot, x3)
    x9 = mapply(x8, x7)
    O = fill(I, x4, x9)
    return O


def solve_6b9890af(I):
    x1 = objects(I, T, T, T)
    x2 = ofcolor(I, 2)
    x3 = argmin(x1, size)
    x4 = subgrid(x2, I)
    x5 = width(x4)
    x6 = divide(x5, 3)
    x7 = upscale(x3, x6)
    x8 = normalize(x7)
    x9 = shift(x8, UNITY)
    O = paint(x4, x9)
    return O


def solve_794b24be(I):
    x1 = ofcolor(I, 1)
    x2 = size(x1)
    x3 = decrement(x2)
    x4 = canvas(0, G3x3)
    x5 = tojvec(x3)
    x6 = connect(ORIGIN, x5)
    x7 = equality(x2, 4)
    x8 = insert(UNITY, x6)
    x9 = branch(x7, x8, x6)
    O = fill(x4, 2, x9)
    return O


def solve_88a10436(I):
    x1 = objects(I, F, F, T)
    x2 = colorfilter(x1, 5)
    x3 = first(x2)
    x4 = center(x3)
    x5 = difference(x1, x2)
    x6 = first(x5)
    x7 = normalize(x6)
    x8 = shift(x7, x4)
    x9 = shift(x8, NEG_UNITY)
    O = paint(I, x9)
    return O


def solve_88a62173(I):
    x1 = lefthalf(I)
    x2 = righthalf(I)
    x3 = tophalf(x1)
    x4 = tophalf(x2)
    x5 = bottomhalf(x1)
    x6 = bottomhalf(x2)
    x7 = astuple(x3, x4)
    x8 = astuple(x5, x6)
    x9 = combine(x7, x8)
    O = leastcommon(x9)
    return O


def solve_890034e9(I):
    x1 = leastcolor(I)
    x2 = ofcolor(I, x1)
    x3 = inbox(x2)
    x4 = recolor(0, x3)
    x5 = occurrences(I, x4)
    x6 = normalize(x2)
    x7 = shift(x6, NEG_UNITY)
    x8 = lbind(shift, x7)
    x9 = mapply(x8, x5)
    O = fill(I, x1, x9)
    return O


def solve_99b1bc43(I):
    x1 = tophalf(I)
    x2 = bottomhalf(I)
    x3 = ofcolor(x1, 0)
    x4 = ofcolor(x2, 0)
    x5 = combine(x3, x4)
    x6 = intersection(x3, x4)
    x7 = difference(x5, x6)
    x8 = shape(x1)
    x9 = canvas(0, x8)
    O = fill(x9, 3, x7)
    return O


def solve_a9f96cdd(I):
    x1 = ofcolor(I, 2)
    x2 = replace(I, 2, 0)
    x3 = shift(x1, NEG_UNITY)
    x4 = fill(x2, 3, x3)
    x5 = shift(x1, UP_RIGHT)
    x6 = fill(x4, 6, x5)
    x7 = shift(x1, DOWN_LEFT)
    x8 = fill(x6, 8, x7)
    x9 = shift(x1, UNITY)
    O = fill(x8, 7, x9)
    return O


def solve_af902bf9(I):
    x1 = ofcolor(I, 4)
    x2 = prapply(connect, x1, x1)
    x3 = fork(either, vline, hline)
    x4 = mfilter(x2, x3)
    x5 = underfill(I,  -1, x4)
    x6 = objects(x5, F, F, T)
    x7 = compose(backdrop, inbox)
    x8 = mapply(x7, x6)
    x9 = fill(x5, 2, x8)
    O = replace(x9,  -1, 0)
    return O


def solve_b548a754(I):
    x1 = objects(I, T, F, T)
    x2 = replace(I, 8, 0)
    x3 = leastcolor(x2)
    x4 = replace(x2, x3, 0)
    x5 = leastcolor(x4)
    x6 = merge(x1)
    x7 = backdrop(x6)
    x8 = box(x6)
    x9 = fill(I, x3, x7)
    O = fill(x9, x5, x8)
    return O


def solve_bdad9b1f(I):
    x1 = ofcolor(I, 2)
    x2 = ofcolor(I, 8)
    x3 = center(x1)
    x4 = center(x2)
    x5 = hfrontier(x3)
    x6 = vfrontier(x4)
    x7 = intersection(x5, x6)
    x8 = fill(I, 2, x5)
    x9 = fill(x8, 8, x6)
    O = fill(x9, 4, x7)
    return O


def solve_c3e719e8(I):
    x1 = mostcolor(I)
    x2 = hconcat(I, I)
    x3 = upscale(I, 3)
    x4 = ofcolor(x3, x1)
    x5 = asindices(x3)
    x6 = difference(x5, x4)
    x7 = hconcat(x2, I)
    x8 = vconcat(x7, x7)
    x9 = vconcat(x8, x7)
    O = fill(x9, 0, x6)
    return O


def solve_de1cd16c(I):
    x1 = leastcolor(I)
    x2 = objects(I, T, F, F)
    x3 = sizefilter(x2, 1)
    x4 = difference(x2, x3)
    x5 = rbind(subgrid, I)
    x6 = apply(x5, x4)
    x7 = rbind(colorcount, x1)
    x8 = argmax(x6, x7)
    x9 = mostcolor(x8)
    O = canvas(x9, UNITY)
    return O


def solve_d8c310e9(I):
    x1 = objects(I, F, F, T)
    x2 = first(x1)
    x3 = hperiod(x2)
    x4 = multiply(x3, 3)
    x5 = tojvec(x3)
    x6 = tojvec(x4)
    x7 = shift(x2, x5)
    x8 = shift(x2, x6)
    x9 = paint(I, x7)
    O = paint(x9, x8)
    return O


def solve_a3325580(I):
    x1 = objects(I, T, F, T)
    x2 = valmax(x1, size)
    x3 = sizefilter(x1, x2)
    x4 = order(x3, leftmost)
    x5 = apply(color, x4)
    x6 = astuple(1, x2)
    x7 = rbind(canvas, x6)
    x8 = apply(x7, x5)
    x9 = merge(x8)
    O = dmirror(x9)
    return O


def solve_8eb1be9a(I):
    x1 = objects(I, T, T, T)
    x2 = first(x1)
    x3 = interval(-2, 4, 1)
    x4 = lbind(shift, x2)
    x5 = height(x2)
    x6 = rbind(multiply, x5)
    x7 = apply(x6, x3)
    x8 = apply(toivec, x7)
    x9 = mapply(x4, x8)
    O = paint(I, x9)
    return O


def solve_321b1fc6(I):
    x1 = objects(I, F, F, T)
    x2 = colorfilter(x1, 8)
    x3 = difference(x1, x2)
    x4 = first(x3)
    x5 = cover(I, x4)
    x6 = normalize(x4)
    x7 = lbind(shift, x6)
    x8 = apply(ulcorner, x2)
    x9 = mapply(x7, x8)
    O = paint(x5, x9)
    return O


def solve_1caeab9d(I):
    x1 = objects(I, T, T, T)
    x2 = ofcolor(I, 1)
    x3 = lowermost(x2)
    x4 = lbind(subtract, x3)
    x5 = chain(toivec, x4, lowermost)
    x6 = fork(shift, identity, x5)
    x7 = merge(x1)
    x8 = cover(I, x7)
    x9 = mapply(x6, x1)
    O = paint(x8, x9)
    return O


def solve_77fdfe62(I):
    x1 = ofcolor(I, 8)
    x2 = subgrid(x1, I)
    x3 = replace(I, 8, 0)
    x4 = replace(x3, 1, 0)
    x5 = compress(x4)
    x6 = width(x2)
    x7 = halve(x6)
    x8 = upscale(x5, x7)
    x9 = ofcolor(x2, 0)
    O = fill(x8, 0, x9)
    return O


def solve_c0f76784(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, 0)
    x3 = sfilter(x2, square)
    x4 = sizefilter(x3, 1)
    x5 = merge(x4)
    x6 = argmax(x3, size)
    x7 = merge(x3)
    x8 = fill(I, 7, x7)
    x9 = fill(x8, 8, x6)
    O = fill(x9, 6, x5)
    return O


def solve_1b60fb0c(I):
    x1 = rot90(I)
    x2 = ofcolor(I, 1)
    x3 = ofcolor(x1, 1)
    x4 = neighbors(ORIGIN)
    x5 = mapply(neighbors, x4)
    x6 = lbind(shift, x3)
    x7 = apply(x6, x5)
    x8 = lbind(intersection, x2)
    x9 = compose(size, x8)
    x10 = argmax(x7, x9)
    O = underfill(I, 2, x10)
    return O


def solve_ddf7fa4f(I):
    x1 = objects(I, T, F, T)
    x2 = sizefilter(x1, 1)
    x3 = colorfilter(x1, 5)
    x4 = product(x2, x3)
    x5 = fork(vmatching, first, last)
    x6 = sfilter(x4, x5)
    x7 = compose(color, first)
    x8 = fork(recolor, x7, last)
    x9 = mapply(x8, x6)
    O = paint(I, x9)
    return O


def solve_47c1f68c(I):
    x1 = leastcolor(I)
    x2 = vmirror(I)
    x3 = objects(I, T, T, T)
    x4 = merge(x3)
    x5 = mostcolor(x4)
    x6 = cellwise(I, x2, x1)
    x7 = hmirror(x6)
    x8 = cellwise(x6, x7, x1)
    x9 = compress(x8)
    O = replace(x9, x1, x5)
    return O


def solve_6c434453(I):
    x1 = objects(I, T, F, T)
    x2 = sizefilter(x1, 8)
    x3 = dneighbors(UNITY)
    x4 = insert(UNITY, x3)
    x5 = merge(x2)
    x6 = cover(I, x5)
    x7 = apply(ulcorner, x2)
    x8 = lbind(shift, x4)
    x9 = mapply(x8, x7)
    O = fill(x6, 2, x9)
    return O


def solve_23581191(I):
    x1 = objects(I, T, T, T)
    x2 = fork(combine, vfrontier, hfrontier)
    x3 = compose(x2, center)
    x4 = fork(recolor, color, x3)
    x5 = mapply(x4, x1)
    x6 = paint(I, x5)
    x7 = fork(intersection, first, last)
    x8 = apply(x3, x1)
    x9 = x7(x8)
    O = fill(x6, 2, x9)
    return O


def solve_c8cbb738(I):
    x1 = mostcolor(I)
    x2 = fgpartition(I)
    x3 = valmax(x2, shape)
    x4 = canvas(x1, x3)
    x5 = apply(normalize, x2)
    x6 = lbind(subtract, x3)
    x7 = chain(halve, x6, shape)
    x8 = fork(shift, identity, x7)
    x9 = mapply(x8, x5)
    O = paint(x4, x9)
    return O


def solve_3eda0437(I):
    x1 = interval(2, 10, 1)
    x2 = prapply(astuple, x1, x1)
    x3 = lbind(canvas, 0)
    x4 = lbind(occurrences, I)
    x5 = lbind(lbind, shift)
    x6 = fork(apply, x5, x4)
    x7 = chain(x6, asobject, x3)
    x8 = mapply(x7, x2)
    x9 = argmax(x8, size)
    O = fill(I, 6, x9)
    return O


def solve_dc0a314f(I):
    x1 = ofcolor(I, 3)
    x2 = replace(I, 3, 0)
    x3 = dmirror(x2)
    x4 = papply(pair, x2, x3)
    x5 = lbind(apply, maximum)
    x6 = apply(x5, x4)
    x7 = cmirror(x6)
    x8 = papply(pair, x6, x7)
    x9 = apply(x5, x8)
    O = subgrid(x1, x9)
    return O


def solve_d4469b4b(I):
    x1 = palette(I)
    x2 = other(x1, 0)
    x3 = equality(x2, 1)
    x4 = equality(x2, 2)
    x5 = branch(x3, UNITY, G2x2)
    x6 = branch(x4, RIGHT, x5)
    x7 = fork(combine, vfrontier, hfrontier)
    x8 = x7(x6)
    x9 = canvas(0, G3x3)
    O = fill(x9, 5, x8)
    return O


def solve_6ecd11f4(I):
    x1 = objects(I, F, T, T)
    x2 = argmax(x1, size)
    x3 = argmin(x1, size)
    x4 = subgrid(x2, I)
    x5 = subgrid(x3, I)
    x6 = width(x4)
    x7 = width(x5)
    x8 = divide(x6, x7)
    x9 = downscale(x4, x8)
    x10 = ofcolor(x9, 0)
    O = fill(x5, 0, x10)
    return O


def solve_760b3cac(I):
    x1 = ofcolor(I, 4)
    x2 = ofcolor(I, 8)
    x3 = ulcorner(x1)
    x4 = index(I, x3)
    x5 = equality(x4, 4)
    x6 = branch(x5,  -1, 1)
    x7 = multiply(x6, 3)
    x8 = tojvec(x7)
    x9 = vmirror(x2)
    x10 = shift(x9, x8)
    O = fill(I, 8, x10)
    return O


def solve_c444b776(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, 0)
    x3 = argmin(x2, size)
    x4 = backdrop(x3)
    x5 = toobject(x4, I)
    x6 = normalize(x5)
    x7 = lbind(shift, x6)
    x8 = compose(x7, ulcorner)
    x9 = mapply(x8, x2)
    O = paint(I, x9)
    return O


def solve_d4a91cb9(I):
    x1 = ofcolor(I, 8)
    x2 = ofcolor(I, 2)
    x3 = first(x1)
    x4 = first(x2)
    x5 = last(x3)
    x6 = first(x4)
    x7 = astuple(x6, x5)
    x8 = connect(x7, x3)
    x9 = connect(x7, x4)
    x10 = combine(x8, x9)
    O = underfill(I, 4, x10)
    return O


def solve_eb281b96(I):
    x1 = height(I)
    x2 = width(I)
    x3 = decrement(x1)
    x4 = astuple(x3, x2)
    x5 = crop(I, ORIGIN, x4)
    x6 = hmirror(x5)
    x7 = vconcat(I, x6)
    x8 = double(x3)
    x9 = astuple(x8, x2)
    x10 = crop(x7, DOWN, x9)
    O = vconcat(x7, x10)
    return O


def solve_ff28f65a(I):
    x1 = objects(I, T, F, T)
    x2 = size(x1)
    x3 = double(x2)
    x4 = interval(0, x3, 2)
    x5 = apply(tojvec, x4)
    x6 = astuple(1, 9)
    x7 = canvas(0, x6)
    x8 = fill(x7, 1, x5)
    x9 = hsplit(x8, 3)
    O = merge(x9)
    return O


def solve_7e0986d6(I):
    x1 = leastcolor(I)
    x2 = ofcolor(I, x1)
    x3 = replace(I, x1, 0)
    x4 = leastcolor(x3)
    x5 = rbind(colorcount, x4)
    x6 = chain(positive, decrement, x5)
    x7 = rbind(toobject, x3)
    x8 = chain(x6, x7, dneighbors)
    x9 = sfilter(x2, x8)
    O = fill(x3, x4, x9)
    return O


def solve_09629e4f(I):
    x1 = objects(I, F, T, T)
    x2 = argmin(x1, numcolors)
    x3 = normalize(x2)
    x4 = upscale(x3, 4)
    x5 = paint(I, x4)
    x6 = ofcolor(I, 5)
    O = fill(x5, 5, x6)
    return O


def solve_a85d4709(I):
    x1 = ofcolor(I, 5)
    x2 = lbind(matcher, last)
    x3 = lbind(sfilter, x1)
    x4 = lbind(mapply, hfrontier)
    x5 = chain(x4, x3, x2)
    x6 = x5(0)
    x7 = x5(2)
    x8 = x5(1)
    x9 = fill(I, 2, x6)
    x10 = fill(x9, 3, x7)
    O = fill(x10, 4, x8)
    return O


def solve_feca6190(I):
    x1 = objects(I, T, F, T)
    x2 = size(x1)
    x3 = multiply(x2, 5)
    x4 = astuple(x3, x3)
    x5 = canvas(0, x4)
    x6 = rbind(shoot, UNITY)
    x7 = compose(x6, center)
    x8 = fork(recolor, color, x7)
    x9 = mapply(x8, x1)
    x10 = paint(x5, x9)
    O = hmirror(x10)
    return O


def solve_a68b268e(I):
    x1 = tophalf(I)
    x2 = bottomhalf(I)
    x3 = lefthalf(x1)
    x4 = righthalf(x1)
    x5 = lefthalf(x2)
    x6 = righthalf(x2)
    x7 = ofcolor(x4, 4)
    x8 = ofcolor(x3, 7)
    x9 = ofcolor(x5, 8)
    x10 = fill(x6, 8, x9)
    x11 = fill(x10, 4, x7)
    O = fill(x11, 7, x8)
    return O


def solve_beb8660c(I):
    x1 = shape(I)
    x2 = objects(I, T, F, T)
    x3 = compose(invert, size)
    x4 = order(x2, x3)
    x5 = apply(normalize, x4)
    x6 = size(x5)
    x7 = interval(0, x6, 1)
    x8 = apply(toivec, x7)
    x9 = mpapply(shift, x5, x8)
    x10 = canvas(0, x1)
    x11 = paint(x10, x9)
    O = rot180(x11)
    return O


def solve_913fb3ed(I):
    x1 = ofcolor(I, 3)
    x2 = ofcolor(I, 8)
    x3 = ofcolor(I, 2)
    x4 = mapply(neighbors, x1)
    x5 = mapply(neighbors, x2)
    x6 = mapply(neighbors, x3)
    x7 = fill(I, 6, x4)
    x8 = fill(x7, 4, x5)
    O = fill(x8, 1, x6)
    return O


def solve_0962bcdd(I):
    x1 = leastcolor(I)
    x2 = replace(I, 0, x1)
    x3 = leastcolor(x2)
    x4 = ofcolor(I, x3)
    x5 = mapply(dneighbors, x4)
    x6 = fill(I, x3, x5)
    x7 = objects(x6, F, T, T)
    x8 = fork(connect, ulcorner, lrcorner)
    x9 = fork(connect, llcorner, urcorner)
    x10 = fork(combine, x8, x9)
    x11 = mapply(x10, x7)
    O = fill(x6, x1, x11)
    return O


def solve_3631a71a(I):
    x1 = shape(I)
    x2 = replace(I, 9, 0)
    x3 = lbind(apply, maximum)
    x4 = dmirror(x2)
    x5 = papply(pair, x2, x4)
    x6 = apply(x3, x5)
    x7 = subtract(x1, G2x2)
    x8 = crop(x6, G2x2, x7)
    x9 = vmirror(x8)
    x10 = objects(x9, T, F, T)
    x11 = merge(x10)
    x12 = shift(x11, G2x2)
    O = paint(x6, x12)
    return O


def solve_05269061(I):
    x1 = objects(I, T, T, T)
    x2 = neighbors(ORIGIN)
    x3 = mapply(neighbors, x2)
    x4 = rbind(multiply, 3)
    x5 = apply(x4, x3)
    x6 = merge(x1)
    x7 = lbind(shift, x6)
    x8 = mapply(x7, x5)
    x9 = shift(x8, UP_RIGHT)
    x10 = shift(x8, DOWN_LEFT)
    x11 = paint(I, x8)
    x12 = paint(x11, x9)
    O = paint(x12, x10)
    return O


def solve_95990924(I):
    x1 = objects(I, T, F, T)
    x2 = apply(outbox, x1)
    x3 = apply(ulcorner, x2)
    x4 = apply(urcorner, x2)
    x5 = apply(llcorner, x2)
    x6 = apply(lrcorner, x2)
    x7 = fill(I, 1, x3)
    x8 = fill(x7, 2, x4)
    x9 = fill(x8, 3, x5)
    O = fill(x9, 4, x6)
    return O


def solve_e509e548(I):
    x1 = objects(I, T, F, T)
    x2 = rbind(subgrid, I)
    x3 = chain(palette, trim, x2)
    x4 = lbind(contained, 3)
    x5 = compose(x4, x3)
    x6 = fork(add, height, width)
    x7 = compose(decrement, x6)
    x8 = fork(equality, size, x7)
    x9 = mfilter(x1, x5)
    x10 = mfilter(x1, x8)
    x11 = replace(I, 3, 6)
    x12 = fill(x11, 2, x9)
    O = fill(x12, 1, x10)
    return O


def solve_d43fd935(I):
    x1 = objects(I, T, F, T)
    x2 = ofcolor(I, 3)
    x3 = sizefilter(x1, 1)
    x4 = rbind(vmatching, x2)
    x5 = rbind(hmatching, x2)
    x6 = fork(either, x4, x5)
    x7 = sfilter(x3, x6)
    x8 = rbind(gravitate, x2)
    x9 = fork(add, center, x8)
    x10 = fork(connect, center, x9)
    x11 = fork(recolor, color, x10)
    x12 = mapply(x11, x7)
    O = paint(I, x12)
    return O


def solve_db3e9e38(I):
    x1 = ofcolor(I, 7)
    x2 = lrcorner(x1)
    x3 = shoot(x2, UP_RIGHT)
    x4 = shoot(x2, NEG_UNITY)
    x5 = combine(x3, x4)
    x6 = rbind(shoot, UP)
    x7 = mapply(x6, x5)
    x8 = last(x2)
    x9 = rbind(subtract, x8)
    x10 = chain(even, x9, last)
    x11 = fill(I, 8, x7)
    x12 = sfilter(x7, x10)
    O = fill(x11, 7, x12)
    return O


def solve_e73095fd(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, 0)
    x3 = fork(equality, toindices, backdrop)
    x4 = sfilter(x2, x3)
    x5 = lbind(mapply, dneighbors)
    x6 = chain(x5, corners, outbox)
    x7 = fork(difference, x6, outbox)
    x8 = ofcolor(I, 5)
    x9 = rbind(intersection, x8)
    x10 = matcher(size, 0)
    x11 = chain(x10, x9, x7)
    x12 = mfilter(x4, x11)
    O = fill(I, 4, x12)
    return O


def solve_1bfc4729(I):
    x1 = asindices(I)
    x2 = tophalf(I)
    x3 = bottomhalf(I)
    x4 = leastcolor(x2)
    x5 = leastcolor(x3)
    x6 = hfrontier(G2x0)
    x7 = box(x1)
    x8 = combine(x6, x7)
    x9 = fill(x2, x4, x8)
    x10 = hmirror(x9)
    x11 = replace(x10, x4, x5)
    O = vconcat(x9, x11)
    return O


def solve_93b581b8(I):
    x1 = fgpartition(I)
    x2 = chain(cmirror, dmirror, merge)
    x3 = x2(x1)
    x4 = upscale(x3, 3)
    x5 = astuple(-2, -2)
    x6 = shift(x4, x5)
    x7 = underpaint(I, x6)
    x8 = toindices(x3)
    x9 = fork(combine, hfrontier, vfrontier)
    x10 = mapply(x9, x8)
    x11 = difference(x10, x8)
    O = fill(x7, 0, x11)
    return O


def solve_9edfc990(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, 0)
    x3 = ofcolor(I, 1)
    x4 = rbind(adjacent, x3)
    x5 = mfilter(x2, x4)
    x6 = recolor(1, x5)
    O = paint(I, x6)
    return O


def solve_a65b410d(I):
    x1 = ofcolor(I, 2)
    x2 = urcorner(x1)
    x3 = shoot(x2, UP_RIGHT)
    x4 = shoot(x2, DOWN_LEFT)
    x5 = underfill(I, 3, x3)
    x6 = underfill(x5, 1, x4)
    x7 = rbind(shoot, LEFT)
    x8 = mapply(x7, x3)
    x9 = mapply(x7, x4)
    x10 = underfill(x6, 1, x9)
    O = underfill(x10, 3, x8)
    return O


def solve_7447852a(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, 0)
    x3 = compose(last, center)
    x4 = order(x2, x3)
    x5 = size(x4)
    x6 = interval(0, x5, 3)
    x7 = rbind(contained, x6)
    x8 = compose(x7, last)
    x9 = interval(0, x5, 1)
    x10 = pair(x4, x9)
    x11 = sfilter(x10, x8)
    x12 = mapply(first, x11)
    O = fill(I, 4, x12)
    return O


def solve_97999447(I):
    x1 = objects(I, T, F, T)
    x2 = apply(toindices, x1)
    x3 = rbind(shoot, RIGHT)
    x4 = compose(x3, center)
    x5 = fork(recolor, color, x4)
    x6 = mapply(x5, x1)
    x7 = paint(I, x6)
    x8 = interval(0, 5, 1)
    x9 = apply(double, x8)
    x10 = apply(increment, x9)
    x11 = apply(tojvec, x10)
    x12 = prapply(shift, x2, x11)
    x13 = merge(x12)
    O = fill(x7, 5, x13)
    return O


def solve_91714a58(I):
    x1 = shape(I)
    x2 = asindices(I)
    x3 = objects(I, T, F, T)
    x4 = argmax(x3, size)
    x5 = mostcolor(x4)
    x6 = canvas(0, x1)
    x7 = paint(x6, x4)
    x8 = rbind(toobject, x7)
    x9 = rbind(colorcount, x5)
    x10 = chain(x9, x8, neighbors)
    x11 = lbind(greater, 3)
    x12 = compose(x11, x10)
    x13 = sfilter(x2, x12)
    O = fill(x7, 0, x13)
    return O


def solve_a61ba2ce(I):
    x1 = objects(I, T, F, T)
    x2 = lbind(index, I)
    x3 = matcher(x2, 0)
    x4 = lbind(extract, x1)
    x5 = rbind(subgrid, I)
    x6 = lbind(compose, x3)
    x7 = chain(x5, x4, x6)
    x8 = x7(ulcorner)
    x9 = x7(urcorner)
    x10 = x7(llcorner)
    x11 = x7(lrcorner)
    x12 = hconcat(x11, x10)
    x13 = hconcat(x9, x8)
    O = vconcat(x12, x13)
    return O


def solve_8e1813be(I):
    x1 = replace(I, 5, 0)
    x2 = objects(x1, T, T, T)
    x3 = first(x2)
    x4 = vline(x3)
    x5 = branch(x4, dmirror, identity)
    x6 = x5(x1)
    x7 = objects(x6, T, T, T)
    x8 = order(x7, uppermost)
    x9 = apply(color, x8)
    x10 = dedupe(x9)
    x11 = size(x10)
    x12 = rbind(repeat, x11)
    x13 = apply(x12, x10)
    O = x5(x13)
    return O


def solve_bc1d5164(I):
    x1 = leastcolor(I)
    x2 = crop(I, ORIGIN, G3x3)
    x3 = crop(I, G2x0, G3x3)
    x4 = tojvec(4)
    x5 = crop(I, x4, G3x3)
    x6 = astuple(2, 4)
    x7 = crop(I, x6, G3x3)
    x8 = canvas(0, G3x3)
    x9 = rbind(ofcolor, x1)
    x10 = astuple(x2, x3)
    x11 = astuple(x5, x7)
    x12 = combine(x10, x11)
    x13 = mapply(x9, x12)
    O = fill(x8, x1, x13)
    return O


def solve_ce602527(I):
    x1 = vmirror(I)
    x2 = fgpartition(x1)
    x3 = order(x2, size)
    x4 = last(x3)
    x5 = remove(x4, x3)
    x6 = compose(toindices, normalize)
    x7 = rbind(upscale, 2)
    x8 = chain(toindices, x7, normalize)
    x9 = x6(x4)
    x10 = rbind(intersection, x9)
    x11 = chain(size, x10, x8)
    x12 = argmax(x5, x11)
    x13 = subgrid(x12, x1)
    O = vmirror(x13)
    return O


def solve_5c2c9af4(I):
    x1 = leastcolor(I)
    x2 = ofcolor(I, x1)
    x3 = center(x2)
    x4 = ulcorner(x2)
    x5 = subtract(x3, x4)
    x6 = multiply( -1, 9)
    x7 = interval(0, 9, 1)
    x8 = interval(0, x6,  -1)
    x9 = lbind(multiply, x5)
    x10 = apply(x9, x7)
    x11 = apply(x9, x8)
    x12 = pair(x10, x11)
    x13 = mapply(box, x12)
    x14 = shift(x13, x3)
    O = fill(I, x1, x14)
    return O


def solve_75b8110e(I):
    x1 = lefthalf(I)
    x2 = righthalf(I)
    x3 = tophalf(x1)
    x4 = bottomhalf(x1)
    x5 = tophalf(x2)
    x6 = bottomhalf(x2)
    x7 = rbind(ofcolor, 0)
    x8 = fork(difference, asindices, x7)
    x9 = fork(toobject, x8, identity)
    x10 = x9(x5)
    x11 = x9(x4)
    x12 = x9(x6)
    x13 = paint(x3, x12)
    x14 = paint(x13, x11)
    O = paint(x14, x10)
    return O


def solve_941d9a10(I):
    x1 = shape(I)
    x2 = objects(I, T, F, F)
    x3 = colorfilter(x2, 0)
    x4 = apply(toindices, x3)
    x5 = lbind(lbind, contained)
    x6 = lbind(extract, x4)
    x7 = compose(x6, x5)
    x8 = decrement(x1)
    x9 = astuple(5, 5)
    x10 = x7(ORIGIN)
    x11 = x7(x8)
    x12 = x7(x9)
    x13 = fill(I, 1, x10)
    x14 = fill(x13, 3, x11)
    O = fill(x14, 2, x12)
    return O


def solve_c3f564a4(I):
    x1 = asindices(I)
    x2 = dmirror(I)
    x3 = invert(9)
    x4 = papply(pair, I, x2)
    x5 = lbind(apply, maximum)
    x6 = apply(x5, x4)
    x7 = ofcolor(x6, 0)
    x8 = difference(x1, x7)
    x9 = toobject(x8, x6)
    x10 = interval(x3, 9, 1)
    x11 = interval(9, x3,  -1)
    x12 = pair(x10, x11)
    x13 = lbind(shift, x9)
    x14 = mapply(x13, x12)
    O = paint(x6, x14)
    return O


def solve_1a07d186(I):
    x1 = objects(I, T, F, T)
    x2 = sizefilter(x1, 1)
    x3 = difference(x1, x2)
    x4 = apply(color, x3)
    x5 = rbind(contained, x4)
    x6 = compose(x5, color)
    x7 = sfilter(x2, x6)
    x8 = lbind(colorfilter, x3)
    x9 = chain(first, x8, color)
    x10 = fork(gravitate, identity, x9)
    x11 = fork(shift, identity, x10)
    x12 = mapply(x11, x7)
    x13 = merge(x2)
    x14 = cover(I, x13)
    O = paint(x14, x12)
    return O


def solve_d687bc17(I):
    x1 = objects(I, T, F, T)
    x2 = sizefilter(x1, 1)
    x3 = difference(x1, x2)
    x4 = apply(color, x3)
    x5 = rbind(contained, x4)
    x6 = compose(x5, color)
    x7 = sfilter(x2, x6)
    x8 = lbind(colorfilter, x3)
    x9 = chain(first, x8, color)
    x10 = fork(gravitate, identity, x9)
    x11 = fork(shift, identity, x10)
    x12 = merge(x2)
    x13 = mapply(x11, x7)
    x14 = cover(I, x12)
    O = paint(x14, x13)
    return O


def solve_9af7a82c(I):
    x1 = objects(I, T, F, F)
    x2 = order(x1, size)
    x3 = valmax(x1, size)
    x4 = rbind(astuple, 1)
    x5 = lbind(subtract, x3)
    x6 = compose(x4, size)
    x7 = chain(x4, x5, size)
    x8 = fork(canvas, color, x6)
    x9 = lbind(canvas, 0)
    x10 = compose(x9, x7)
    x11 = fork(vconcat, x8, x10)
    x12 = compose(cmirror, x11)
    x13 = apply(x12, x2)
    x14 = merge(x13)
    O = cmirror(x14)
    return O


def solve_6e19193c(I):
    x1 = leastcolor(I)
    x2 = objects(I, T, F, T)
    x3 = rbind(toobject, I)
    x4 = compose(first, delta)
    x5 = rbind(colorcount, x1)
    x6 = matcher(x5, 2)
    x7 = chain(x6, x3, dneighbors)
    x8 = rbind(sfilter, x7)
    x9 = chain(first, x8, toindices)
    x10 = fork(subtract, x4, x9)
    x11 = fork(shoot, x4, x10)
    x12 = mapply(x11, x2)
    x13 = fill(I, x1, x12)
    x14 = mapply(delta, x2)
    O = fill(x13, 0, x14)
    return O


def solve_ef135b50(I):
    x1 = ofcolor(I, 2)
    x2 = ofcolor(I, 0)
    x3 = product(x1, x1)
    x4 = power(first, 2)
    x5 = compose(first, last)
    x6 = fork(equality, x4, x5)
    x7 = sfilter(x3, x6)
    x8 = fork(connect, first, last)
    x9 = mapply(x8, x7)
    x10 = intersection(x9, x2)
    x11 = fill(I, 9, x10)
    x12 = trim(x11)
    x13 = asobject(x12)
    x14 = shift(x13, UNITY)
    O = paint(I, x14)
    return O


def solve_cbded52d(I):
    x1 = objects(I, T, F, T)
    x2 = sizefilter(x1, 1)
    x3 = product(x2, x2)
    x4 = fork(vmatching, first, last)
    x5 = fork(hmatching, first, last)
    x6 = fork(either, x4, x5)
    x7 = sfilter(x3, x6)
    x8 = compose(center, first)
    x9 = compose(center, last)
    x10 = fork(connect, x8, x9)
    x11 = chain(initset, center, x10)
    x12 = compose(color, first)
    x13 = fork(recolor, x12, x11)
    x14 = mapply(x13, x7)
    O = paint(I, x14)
    return O


def solve_8a004b2b(I):
    x1 = objects(I, F, T, T)
    x2 = ofcolor(I, 4)
    x3 = subgrid(x2, I)
    x4 = argmax(x1, lowermost)
    x5 = normalize(x4)
    x6 = replace(x3, 4, 0)
    x7 = objects(x6, T, F, T)
    x8 = merge(x7)
    x9 = width(x8)
    x10 = ulcorner(x8)
    x11 = width(x4)
    x12 = divide(x9, x11)
    x13 = upscale(x5, x12)
    x14 = shift(x13, x10)
    O = paint(x3, x14)
    return O


def solve_e26a3af2(I):
    x1 = rot90(I)
    x2 = apply(mostcommon, I)
    x3 = apply(mostcommon, x1)
    x4 = repeat(x2, 1)
    x5 = repeat(x3, 1)
    x6 = compose(size, dedupe)
    x7 = x6(x2)
    x8 = x6(x3)
    x9 = greater(x8, x7)
    x10 = branch(x9, height, width)
    x11 = x10(I)
    x12 = rot90(x4)
    x13 = branch(x9, x5, x12)
    x14 = branch(x9, vupscale, hupscale)
    O = x14(x13, x11)
    return O


def solve_6cf79266(I):
    x1 = ofcolor(I, 0)
    x2 = astuple(0, ORIGIN)
    x3 = initset(x2)
    x4 = upscale(x3, 3)
    x5 = toindices(x4)
    x6 = lbind(shift, x5)
    x7 = rbind(difference, x1)
    x8 = chain(size, x7, x6)
    x9 = matcher(x8, 0)
    x10 = lbind(add, NEG_UNITY)
    x11 = chain(flip, x9, x10)
    x12 = fork(both, x9, x11)
    x13 = sfilter(x1, x12)
    x14 = mapply(x6, x13)
    O = fill(I, 1, x14)
    return O


def solve_a87f7484(I):
    x1 = numcolors(I)
    x2 = dmirror(I)
    x3 = portrait(I)
    x4 = branch(x3, dmirror, identity)
    x5 = x4(I)
    x6 = decrement(x1)
    x7 = hsplit(x5, x6)
    x8 = rbind(ofcolor, 0)
    x9 = apply(x8, x7)
    x10 = leastcommon(x9)
    x11 = matcher(x8, x10)
    x12 = extract(x7, x11)
    O = x4(x12)
    return O


def solve_4093f84a(I):
    x1 = leastcolor(I)
    x2 = replace(I, x1, 5)
    x3 = ofcolor(I, 5)
    x4 = portrait(x3)
    x5 = branch(x4, identity, dmirror)
    x6 = x5(x2)
    x7 = lefthalf(x6)
    x8 = righthalf(x6)
    x9 = rbind(order, identity)
    x10 = rbind(order, invert)
    x11 = apply(x9, x7)
    x12 = apply(x10, x8)
    x13 = hconcat(x11, x12)
    O = x5(x13)
    return O


def solve_ba26e723(I):
    x1 = rbind(divide, 3)
    x2 = rbind(multiply, 3)
    x3 = compose(x2, x1)
    x4 = fork(equality, identity, x3)
    x5 = compose(x4, last)
    x6 = ofcolor(I, 4)
    x7 = sfilter(x6, x5)
    O = fill(I, 6, x7)
    return O


def solve_4612dd53(I):
    x1 = ofcolor(I, 1)
    x2 = box(x1)
    x3 = fill(I, 2, x2)
    x4 = subgrid(x1, x3)
    x5 = ofcolor(x4, 1)
    x6 = mapply(vfrontier, x5)
    x7 = mapply(hfrontier, x5)
    x8 = size(x6)
    x9 = size(x7)
    x10 = greater(x8, x9)
    x11 = branch(x10, x7, x6)
    x12 = fill(x4, 2, x11)
    x13 = ofcolor(x12, 2)
    x14 = ulcorner(x1)
    x15 = shift(x13, x14)
    O = underfill(I, 2, x15)
    return O


def solve_29c11459(I):
    x1 = lefthalf(I)
    x2 = righthalf(I)
    x3 = objects(x2, T, F, T)
    x4 = objects(x1, T, F, T)
    x5 = compose(hfrontier, center)
    x6 = fork(recolor, color, x5)
    x7 = mapply(x6, x4)
    x8 = paint(x1, x7)
    x9 = mapply(x6, x3)
    x10 = paint(I, x9)
    x11 = objects(x8, T, F, T)
    x12 = apply(urcorner, x11)
    x13 = shift(x12, RIGHT)
    x14 = merge(x11)
    x15 = paint(x10, x14)
    O = fill(x15, 5, x13)
    return O


def solve_963e52fc(I):
    x1 = width(I)
    x2 = asobject(I)
    x3 = hperiod(x2)
    x4 = height(x2)
    x5 = astuple(x4, x3)
    x6 = ulcorner(x2)
    x7 = crop(I, x6, x5)
    x8 = rot90(x7)
    x9 = double(x1)
    x10 = divide(x9, x3)
    x11 = increment(x10)
    x12 = repeat(x8, x11)
    x13 = merge(x12)
    x14 = rot270(x13)
    x15 = astuple(x4, x9)
    O = crop(x14, ORIGIN, x15)
    return O


def solve_ae3edfdc(I):
    x1 = objects(I, T, F, T)
    x2 = replace(I, 3, 0)
    x3 = replace(x2, 7, 0)
    x4 = lbind(colorfilter, x1)
    x5 = lbind(rbind, gravitate)
    x6 = chain(x5, first, x4)
    x7 = x6(2)
    x8 = x6(1)
    x9 = x4(3)
    x10 = x4(7)
    x11 = fork(shift, identity, x7)
    x12 = fork(shift, identity, x8)
    x13 = mapply(x11, x9)
    x14 = mapply(x12, x10)
    x15 = paint(x3, x13)
    O = paint(x15, x14)
    return O


def solve_1f0c79e5(I):
    x1 = ofcolor(I, 2)
    x2 = replace(I, 2, 0)
    x3 = leastcolor(x2)
    x4 = ofcolor(x2, x3)
    x5 = combine(x1, x4)
    x6 = recolor(x3, x5)
    x7 = compose(decrement, double)
    x8 = ulcorner(x5)
    x9 = invert(x8)
    x10 = shift(x1, x9)
    x11 = apply(x7, x10)
    x12 = interval(0, 9, 1)
    x13 = prapply(multiply, x11, x12)
    x14 = lbind(shift, x6)
    x15 = mapply(x14, x13)
    O = paint(I, x15)
    return O


def solve_56dc2b01(I):
    x1 = objects(I, T, F, T)
    x2 = colorfilter(x1, 3)
    x3 = first(x2)
    x4 = ofcolor(I, 2)
    x5 = gravitate(x3, x4)
    x6 = first(x5)
    x7 = equality(x6, 0)
    x8 = branch(x7, width, height)
    x9 = x8(x3)
    x10 = gravitate(x4, x3)
    x11 = sign(x10)
    x12 = multiply(x11, x9)
    x13 = crement(x12)
    x14 = recolor(8, x4)
    x15 = shift(x14, x13)
    x16 = paint(I, x15)
    O = move(x16, x3, x5)
    return O


def solve_e48d4e1a(I):
    x1 = shape(I)
    x2 = ofcolor(I, 5)
    x3 = fill(I, 0, x2)
    x4 = leastcolor(x3)
    x5 = size(x2)
    x6 = ofcolor(I, x4)
    x7 = rbind(toobject, I)
    x8 = rbind(colorcount, x4)
    x9 = chain(x8, x7, dneighbors)
    x10 = matcher(x9, 4)
    x11 = extract(x6, x10)
    x12 = multiply(DOWN_LEFT, x5)
    x13 = add(x12, x11)
    x14 = canvas(0, x1)
    x15 = fork(combine, vfrontier, hfrontier)
    x16 = x15(x13)
    O = fill(x14, x4, x16)
    return O


def solve_6773b310(I):
    x1 = compress(I)
    x2 = neighbors(ORIGIN)
    x3 = insert(ORIGIN, x2)
    x4 = rbind(multiply, 3)
    x5 = apply(x4, x3)
    x6 = astuple(4, 4)
    x7 = shift(x5, x6)
    x8 = fork(insert, identity, neighbors)
    x9 = apply(x8, x7)
    x10 = rbind(toobject, x1)
    x11 = apply(x10, x9)
    x12 = rbind(colorcount, 6)
    x13 = matcher(x12, 2)
    x14 = mfilter(x11, x13)
    x15 = fill(x1, 1, x14)
    x16 = replace(x15, 6, 0)
    O = downscale(x16, 3)
    return O


def solve_780d0b14(I):
    x1 = asindices(I)
    x2 = objects(I, T, T, T)
    x3 = rbind(greater, 2)
    x4 = compose(x3, size)
    x5 = sfilter(x2, x4)
    x6 = totuple(x5)
    x7 = apply(color, x6)
    x8 = apply(center, x6)
    x9 = pair(x7, x8)
    x10 = fill(I, 0, x1)
    x11 = paint(x10, x9)
    x12 = rbind(greater, 1)
    x13 = compose(dedupe, totuple)
    x14 = chain(x12, size, x13)
    x15 = sfilter(x11, x14)
    x16 = rot90(x15)
    x17 = sfilter(x16, x14)
    O = rot270(x17)
    return O


def solve_2204b7a8(I):
    x1 = objects(I, T, F, T)
    x2 = lbind(sfilter, x1)
    x3 = compose(size, x2)
    x4 = x3(vline)
    x5 = x3(hline)
    x6 = greater(x4, x5)
    x7 = branch(x6, lefthalf, tophalf)
    x8 = branch(x6, righthalf, bottomhalf)
    x9 = branch(x6, hconcat, vconcat)
    x10 = x7(I)
    x11 = x8(I)
    x12 = index(x10, ORIGIN)
    x13 = shape(x11)
    x14 = decrement(x13)
    x15 = index(x11, x14)
    x16 = replace(x10, 3, x12)
    x17 = replace(x11, 3, x15)
    O = x9(x16, x17)
    return O


def solve_d9f24cd1(I):
    x1 = ofcolor(I, 2)
    x2 = ofcolor(I, 5)
    x3 = prapply(connect, x1, x2)
    x4 = mfilter(x3, vline)
    x5 = underfill(I, 2, x4)
    x6 = matcher(numcolors, 2)
    x7 = objects(x5, F, F, T)
    x8 = sfilter(x7, x6)
    x9 = difference(x7, x8)
    x10 = colorfilter(x9, 2)
    x11 = mapply(toindices, x10)
    x12 = apply(urcorner, x8)
    x13 = shift(x12, UNITY)
    x14 = rbind(shoot, UP)
    x15 = mapply(x14, x13)
    x16 = fill(x5, 2, x15)
    x17 = mapply(vfrontier, x11)
    O = fill(x16, 2, x17)
    return O


def solve_b782dc8a(I):
    x1 = leastcolor(I)
    x2 = objects(I, T, F, F)
    x3 = ofcolor(I, x1)
    x4 = first(x3)
    x5 = dneighbors(x4)
    x6 = toobject(x5, I)
    x7 = mostcolor(x6)
    x8 = ofcolor(I, x7)
    x9 = colorfilter(x2, 0)
    x10 = rbind(adjacent, x8)
    x11 = mfilter(x9, x10)
    x12 = toindices(x11)
    x13 = rbind(manhattan, x3)
    x14 = chain(even, x13, initset)
    x15 = sfilter(x12, x14)
    x16 = difference(x12, x15)
    x17 = fill(I, x1, x15)
    O = fill(x17, x7, x16)
    return O


def solve_673ef223(I):
    x1 = objects(I, T, F, T)
    x2 = ofcolor(I, 8)
    x3 = replace(I, 8, 4)
    x4 = colorfilter(x1, 2)
    x5 = argmin(x1, uppermost)
    x6 = apply(uppermost, x4)
    x7 = fork(subtract, maximum, minimum)
    x8 = x7(x6)
    x9 = toivec(x8)
    x10 = leftmost(x5)
    x11 = equality(x10, 0)
    x12 = branch(x11, LEFT, RIGHT)
    x13 = rbind(shoot, x12)
    x14 = mapply(x13, x2)
    x15 = underfill(x3, 8, x14)
    x16 = shift(x2, x9)
    x17 = mapply(hfrontier, x16)
    O = underfill(x15, 8, x17)
    return O


def solve_f5b8619d(I):
    x1 = leastcolor(I)
    x2 = ofcolor(I, x1)
    x3 = mapply(vfrontier, x2)
    x4 = underfill(I, 8, x3)
    x5 = hconcat(x4, x4)
    O = vconcat(x5, x5)
    return O


def solve_f8c80d96(I):
    x1 = leastcolor(I)
    x2 = objects(I, T, F, F)
    x3 = colorfilter(x2, x1)
    x4 = argmax(x3, size)
    x5 = argmin(x2, width)
    x6 = size(x5)
    x7 = equality(x6, 1)
    x8 = branch(x7, identity, outbox)
    x9 = chain(outbox, outbox, x8)
    x10 = power(x9, 2)
    x11 = power(x9, 3)
    x12 = x9(x4)
    x13 = x10(x4)
    x14 = x11(x4)
    x15 = fill(I, x1, x12)
    x16 = fill(x15, x1, x13)
    x17 = fill(x16, x1, x14)
    O = replace(x17, 0, 5)
    return O


def solve_ecdecbb3(I):
    x1 = objects(I, T, F, T)
    x2 = colorfilter(x1, 2)
    x3 = colorfilter(x1, 8)
    x4 = product(x2, x3)
    x5 = fork(gravitate, first, last)
    x6 = compose(crement, x5)
    x7 = compose(center, first)
    x8 = fork(add, x7, x6)
    x9 = fork(connect, x7, x8)
    x10 = apply(x9, x4)
    x11 = lbind(greater, 8)
    x12 = compose(x11, size)
    x13 = mfilter(x10, x12)
    x14 = fill(I, 2, x13)
    x15 = apply(x8, x4)
    x16 = intersection(x13, x15)
    x17 = mapply(neighbors, x16)
    O = fill(x14, 8, x17)
    return O


def solve_e5062a87(I):
    x1 = ofcolor(I, 2)
    x2 = recolor(0, x1)
    x3 = normalize(x2)
    x4 = occurrences(I, x2)
    x5 = lbind(shift, x3)
    x6 = apply(x5, x4)
    x7 = astuple(1, 3)
    x8 = astuple(5, 1)
    x9 = astuple(2, 6)
    x10 = initset(x7)
    x11 = insert(x8, x10)
    x12 = insert(x9, x11)
    x13 = rbind(contained, x12)
    x14 = chain(flip, x13, ulcorner)
    x15 = sfilter(x6, x14)
    x16 = merge(x15)
    x17 = recolor(2, x16)
    O = paint(I, x17)
    return O


def solve_a8d7556c(I):
    x1 = initset(ORIGIN)
    x2 = recolor(0, x1)
    x3 = upscale(x2, 2)
    x4 = occurrences(I, x3)
    x5 = lbind(shift, x3)
    x6 = mapply(x5, x4)
    x7 = fill(I, 2, x6)
    x8 = add(6, 6)
    x9 = astuple(8, x8)
    x10 = index(x7, x9)
    x11 = equality(x10, 2)
    x12 = initset(x9)
    x13 = add(x9, DOWN)
    x14 = insert(x13, x12)
    x15 = toobject(x14, x7)
    x16 = toobject(x14, I)
    x17 = branch(x11, x16, x15)
    O = paint(x7, x17)
    return O


def solve_4938f0c2(I):
    x1 = objects(I, T, T, T)
    x2 = ofcolor(I, 2)
    x3 = vmirror(x2)
    x4 = height(x2)
    x5 = width(x2)
    x6 = toivec(x4)
    x7 = tojvec(x5)
    x8 = add(x7, G0x2)
    x9 = add(x6, G2x0)
    x10 = shift(x3, x8)
    x11 = fill(I, 2, x10)
    x12 = ofcolor(x11, 2)
    x13 = hmirror(x12)
    x14 = shift(x13, x9)
    x15 = fill(x11, 2, x14)
    x16 = size(x1)
    x17 = greater(x16, 4)
    O = branch(x17, I, x15)
    return O


def solve_834ec97d(I):
    x1 = asindices(I)
    x2 = objects(I, T, F, T)
    x3 = first(x2)
    x4 = shift(x3, DOWN)
    x5 = fill(I, 0, x3)
    x6 = paint(x5, x4)
    x7 = uppermost(x4)
    x8 = leftmost(x4)
    x9 = subtract(x8, 10)
    x10 = add(x8, 10)
    x11 = interval(x9, x10, 2)
    x12 = lbind(greater, x7)
    x13 = compose(x12, first)
    x14 = rbind(contained, x11)
    x15 = compose(x14, last)
    x16 = sfilter(x1, x13)
    x17 = sfilter(x16, x15)
    O = fill(x6, 4, x17)
    return O


def solve_846bdb03(I):
    x1 = objects(I, F, F, T)
    x2 = rbind(colorcount, 4)
    x3 = matcher(x2, 0)
    x4 = extract(x1, x3)
    x5 = remove(x4, x1)
    x6 = merge(x5)
    x7 = subgrid(x6, I)
    x8 = index(x7, DOWN)
    x9 = subgrid(x4, I)
    x10 = lefthalf(x9)
    x11 = palette(x10)
    x12 = other(x11, 0)
    x13 = equality(x8, x12)
    x14 = branch(x13, identity, vmirror)
    x15 = x14(x4)
    x16 = normalize(x15)
    x17 = shift(x16, UNITY)
    O = paint(x7, x17)
    return O


def solve_90f3ed37(I):
    x1 = objects(I, T, T, T)
    x2 = order(x1, uppermost)
    x3 = first(x2)
    x4 = remove(x3, x2)
    x5 = normalize(x3)
    x6 = lbind(shift, x5)
    x7 = compose(x6, ulcorner)
    x8 = interval(2,  -1,  -1)
    x9 = apply(tojvec, x8)
    x10 = rbind(apply, x9)
    x11 = lbind(compose, size)
    x12 = lbind(lbind, intersection)
    x13 = compose(x11, x12)
    x14 = lbind(lbind, shift)
    x15 = chain(x10, x14, x7)
    x16 = fork(argmax, x15, x13)
    x17 = mapply(x16, x4)
    O = underfill(I, 1, x17)
    return O


def solve_8403a5d5(I):
    x1 = asindices(I)
    x2 = objects(I, T, F, T)
    x3 = first(x2)
    x4 = color(x3)
    x5 = leftmost(x3)
    x6 = interval(x5, 10, 2)
    x7 = rbind(contained, x6)
    x8 = compose(x7, last)
    x9 = sfilter(x1, x8)
    x10 = increment(x5)
    x11 = add(x5, 3)
    x12 = interval(x10, 10, 4)
    x13 = interval(x11, 10, 4)
    x14 = lbind(astuple, 9)
    x15 = apply(tojvec, x12)
    x16 = apply(x14, x13)
    x17 = fill(I, x4, x9)
    x18 = fill(x17, 5, x15)
    O = fill(x18, 5, x16)
    return O


def solve_91413438(I):
    x1 = colorcount(I, 0)
    x2 = subtract(9, x1)
    x3 = multiply(x1, 3)
    x4 = multiply(x3, x1)
    x5 = subtract(x4, 3)
    x6 = astuple(3, x5)
    x7 = canvas(0, x6)
    x8 = hconcat(I, x7)
    x9 = objects(x8, T, T, T)
    x10 = first(x9)
    x11 = lbind(shift, x10)
    x12 = compose(x11, tojvec)
    x13 = interval(0, x2, 1)
    x14 = rbind(multiply, 3)
    x15 = apply(x14, x13)
    x16 = mapply(x12, x15)
    x17 = paint(x8, x16)
    x18 = hsplit(x17, x1)
    O = merge(x18)
    return O


def solve_539a4f51(I):
    x1 = shape(I)
    x2 = index(I, ORIGIN)
    x3 = colorcount(I, 0)
    x4 = decrement(x1)
    x5 = positive(x3)
    x6 = branch(x5, x4, x1)
    x7 = crop(I, ORIGIN, x6)
    x8 = width(x7)
    x9 = astuple(1, x8)
    x10 = crop(x7, ORIGIN, x9)
    x11 = vupscale(x10, x8)
    x12 = dmirror(x11)
    x13 = hconcat(x7, x11)
    x14 = hconcat(x12, x7)
    x15 = vconcat(x13, x14)
    x16 = asobject(x15)
    x17 = multiply(UNITY, 10)
    x18 = canvas(x2, x17)
    O = paint(x18, x16)
    return O


def solve_5daaa586(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, 0)
    x3 = rbind(bordering, I)
    x4 = compose(flip, x3)
    x5 = extract(x2, x4)
    x6 = outbox(x5)
    x7 = subgrid(x6, I)
    x8 = fgpartition(x7)
    x9 = argmax(x8, size)
    x10 = color(x9)
    x11 = toindices(x9)
    x12 = prapply(connect, x11, x11)
    x13 = mfilter(x12, vline)
    x14 = mfilter(x12, hline)
    x15 = size(x13)
    x16 = size(x14)
    x17 = greater(x15, x16)
    x18 = branch(x17, x13, x14)
    O = fill(x7, x10, x18)
    return O


def solve_3bdb4ada(I):
    x1 = objects(I, T, F, T)
    x2 = totuple(x1)
    x3 = compose(increment, ulcorner)
    x4 = compose(decrement, lrcorner)
    x5 = apply(x3, x2)
    x6 = apply(x4, x2)
    x7 = papply(connect, x5, x6)
    x8 = apply(last, x5)
    x9 = compose(last, first)
    x10 = power(last, 2)
    x11 = fork(subtract, x9, x10)
    x12 = compose(even, x11)
    x13 = lbind(rbind, astuple)
    x14 = lbind(compose, x12)
    x15 = compose(x14, x13)
    x16 = fork(sfilter, first, x15)
    x17 = pair(x7, x8)
    x18 = mapply(x16, x17)
    O = fill(I, 0, x18)
    return O


def solve_ec883f72(I):
    x1 = palette(I)
    x2 = objects(I, T, T, T)
    x3 = fork(multiply, height, width)
    x4 = argmax(x2, x3)
    x5 = color(x4)
    x6 = remove(0, x1)
    x7 = other(x6, x5)
    x8 = lrcorner(x4)
    x9 = llcorner(x4)
    x10 = urcorner(x4)
    x11 = ulcorner(x4)
    x12 = shoot(x8, UNITY)
    x13 = shoot(x9, DOWN_LEFT)
    x14 = shoot(x10, UP_RIGHT)
    x15 = shoot(x11, NEG_UNITY)
    x16 = combine(x12, x13)
    x17 = combine(x14, x15)
    x18 = combine(x16, x17)
    O = underfill(I, x7, x18)
    return O


def solve_2bee17df(I):
    x1 = height(I)
    x2 = rot90(I)
    x3 = subtract(x1, 2)
    x4 = interval(0, x1, 1)
    x5 = rbind(colorcount, 0)
    x6 = matcher(x5, x3)
    x7 = rbind(vsplit, x1)
    x8 = lbind(apply, x6)
    x9 = compose(x8, x7)
    x10 = x9(I)
    x11 = pair(x4, x10)
    x12 = sfilter(x11, last)
    x13 = mapply(hfrontier, x12)
    x14 = x9(x2)
    x15 = pair(x14, x4)
    x16 = sfilter(x15, first)
    x17 = mapply(vfrontier, x16)
    x18 = astuple(x13, x17)
    x19 = merge(x18)
    O = underfill(I, 3, x19)
    return O


def solve_e8dc4411(I):
    x1 = leastcolor(I)
    x2 = ofcolor(I, 0)
    x3 = ofcolor(I, x1)
    x4 = position(x2, x3)
    x5 = fork(connect, ulcorner, lrcorner)
    x6 = x5(x2)
    x7 = intersection(x2, x6)
    x8 = equality(x6, x7)
    x9 = fork(subtract, identity, crement)
    x10 = fork(add, identity, x9)
    x11 = branch(x8, identity, x10)
    x12 = shape(x2)
    x13 = multiply(x12, x4)
    x14 = apply(x11, x13)
    x15 = interval(1, 5, 1)
    x16 = lbind(multiply, x14)
    x17 = apply(x16, x15)
    x18 = lbind(shift, x2)
    x19 = mapply(x18, x17)
    O = fill(I, x1, x19)
    return O


def solve_e40b9e2f(I):
    x1 = objects(I, F, T, T)
    x2 = neighbors(ORIGIN)
    x3 = mapply(neighbors, x2)
    x4 = first(x1)
    x5 = lbind(intersection, x4)
    x6 = compose(hmirror, vmirror)
    x7 = x6(x4)
    x8 = lbind(shift, x7)
    x9 = apply(x8, x3)
    x10 = argmax(x9, x5)
    x11 = paint(I, x10)
    x12 = objects(x11, F, T, T)
    x13 = first(x12)
    x14 = compose(size, x5)
    x15 = compose(vmirror, dmirror)
    x16 = x15(x13)
    x17 = lbind(shift, x16)
    x18 = apply(x17, x3)
    x19 = argmax(x18, x14)
    O = paint(x11, x19)
    return O


def solve_29623171(I):
    x1 = leastcolor(I)
    x2 = interval(0, 9, 4)
    x3 = product(x2, x2)
    x4 = rbind(add, 3)
    x5 = rbind(interval, 1)
    x6 = fork(x5, identity, x4)
    x7 = compose(x6, first)
    x8 = compose(x6, last)
    x9 = fork(product, x7, x8)
    x10 = rbind(colorcount, x1)
    x11 = rbind(toobject, I)
    x12 = compose(x10, x11)
    x13 = apply(x9, x3)
    x14 = valmax(x13, x12)
    x15 = matcher(x12, x14)
    x16 = compose(flip, x15)
    x17 = mfilter(x13, x15)
    x18 = mfilter(x13, x16)
    x19 = fill(I, x1, x17)
    O = fill(x19, 0, x18)
    return O


def solve_a2fd1cf0(I):
    x1 = ofcolor(I, 2)
    x2 = ofcolor(I, 3)
    x3 = uppermost(x1)
    x4 = leftmost(x1)
    x5 = uppermost(x2)
    x6 = leftmost(x2)
    x7 = astuple(x3, x5)
    x8 = minimum(x7)
    x9 = maximum(x7)
    x10 = astuple(x8, x6)
    x11 = astuple(x9, x6)
    x12 = connect(x10, x11)
    x13 = astuple(x4, x6)
    x14 = minimum(x13)
    x15 = maximum(x13)
    x16 = astuple(x3, x14)
    x17 = astuple(x3, x15)
    x18 = connect(x16, x17)
    x19 = combine(x12, x18)
    O = underfill(I, 8, x19)
    return O


def solve_b0c4d837(I):
    x1 = ofcolor(I, 5)
    x2 = ofcolor(I, 8)
    x3 = height(x1)
    x4 = decrement(x3)
    x5 = height(x2)
    x6 = subtract(x4, x5)
    x7 = astuple(1, x6)
    x8 = canvas(8, x7)
    x9 = subtract(6, x6)
    x10 = astuple(1, x9)
    x11 = canvas(0, x10)
    x12 = hconcat(x8, x11)
    x13 = hsplit(x12, 2)
    x14 = first(x13)
    x15 = last(x13)
    x16 = vmirror(x15)
    x17 = vconcat(x14, x16)
    x18 = astuple(1, 3)
    x19 = canvas(0, x18)
    O = vconcat(x17, x19)
    return O


def solve_8731374e(I):
    x1 = objects(I, T, F, F)
    x2 = argmax(x1, size)
    x3 = subgrid(x2, I)
    x4 = height(x3)
    x5 = width(x3)
    x6 = vsplit(x3, x4)
    x7 = lbind(greater, 4)
    x8 = compose(x7, numcolors)
    x9 = sfilter(x6, x8)
    x10 = merge(x9)
    x11 = rot90(x10)
    x12 = vsplit(x11, x5)
    x13 = sfilter(x12, x8)
    x14 = merge(x13)
    x15 = rot270(x14)
    x16 = leastcolor(x15)
    x17 = ofcolor(x15, x16)
    x18 = fork(combine, vfrontier, hfrontier)
    x19 = mapply(x18, x17)
    O = fill(x15, x16, x19)
    return O


def solve_272f95fa(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, 0)
    x3 = apply(toindices, x2)
    x4 = rbind(bordering, I)
    x5 = compose(flip, x4)
    x6 = extract(x3, x5)
    x7 = remove(x6, x3)
    x8 = lbind(vmatching, x6)
    x9 = lbind(hmatching, x6)
    x10 = sfilter(x7, x8)
    x11 = sfilter(x7, x9)
    x12 = argmin(x10, uppermost)
    x13 = argmax(x10, uppermost)
    x14 = argmin(x11, leftmost)
    x15 = argmax(x11, leftmost)
    x16 = fill(I, 6, x6)
    x17 = fill(x16, 2, x12)
    x18 = fill(x17, 1, x13)
    x19 = fill(x18, 4, x14)
    O = fill(x19, 3, x15)
    return O


def solve_db93a21d(I):
    x1 = objects(I, T, T, T)
    x2 = ofcolor(I, 9)
    x3 = colorfilter(x1, 9)
    x4 = rbind(shoot, DOWN)
    x5 = mapply(x4, x2)
    x6 = underfill(I, 1, x5)
    x7 = compose(halve, width)
    x8 = rbind(greater, 1)
    x9 = compose(x8, x7)
    x10 = matcher(x7, 3)
    x11 = power(outbox, 2)
    x12 = power(outbox, 3)
    x13 = mapply(outbox, x3)
    x14 = sfilter(x3, x9)
    x15 = sfilter(x3, x10)
    x16 = mapply(x11, x14)
    x17 = mapply(x12, x15)
    x18 = fill(x6, 3, x13)
    x19 = fill(x18, 3, x16)
    O = fill(x19, 3, x17)
    return O


def solve_53b68214(I):
    x1 = width(I)
    x2 = objects(I, T, T, T)
    x3 = first(x2)
    x4 = vperiod(x3)
    x5 = toivec(x4)
    x6 = interval(0, 9, 1)
    x7 = lbind(multiply, x5)
    x8 = apply(x7, x6)
    x9 = lbind(shift, x3)
    x10 = mapply(x9, x8)
    x11 = astuple(x1, x1)
    x12 = portrait(x3)
    x13 = shape(x3)
    x14 = add(DOWN, x13)
    x15 = decrement(x14)
    x16 = shift(x3, x15)
    x17 = branch(x12, x10, x16)
    x18 = canvas(0, x11)
    x19 = paint(x18, x3)
    O = paint(x19, x17)
    return O


def solve_d6ad076f(I):
    x1 = objects(I, T, F, T)
    x2 = argmin(x1, size)
    x3 = argmax(x1, size)
    x4 = vmatching(x2, x3)
    x5 = branch(x4, DOWN, RIGHT)
    x6 = branch(x4, uppermost, leftmost)
    x7 = valmax(x1, x6)
    x8 = x6(x2)
    x9 = equality(x7, x8)
    x10 = branch(x9,  -1, 1)
    x11 = multiply(x5, x10)
    x12 = inbox(x2)
    x13 = rbind(shoot, x11)
    x14 = mapply(x13, x12)
    x15 = underfill(I, 8, x14)
    x16 = objects(x15, T, F, T)
    x17 = colorfilter(x16, 8)
    x18 = rbind(bordering, I)
    x19 = mfilter(x17, x18)
    O = cover(x15, x19)
    return O


def solve_6cdd2623(I):
    x1 = leastcolor(I)
    x2 = ofcolor(I, x1)
    x3 = prapply(connect, x2, x2)
    x4 = fgpartition(I)
    x5 = merge(x4)
    x6 = cover(I, x5)
    x7 = fork(either, hline, vline)
    x8 = box(x5)
    x9 = rbind(difference, x8)
    x10 = chain(positive, size, x9)
    x11 = fork(both, x7, x10)
    x12 = mfilter(x3, x11)
    O = fill(x6, x1, x12)
    return O


def solve_a3df8b1e(I):
    x1 = shape(I)
    x2 = ofcolor(I, 1)
    x3 = first(x2)
    x4 = shoot(x3, UP_RIGHT)
    x5 = fill(I, 1, x4)
    x6 = ofcolor(x5, 1)
    x7 = urcorner(x6)
    x8 = shoot(x7, NEG_UNITY)
    x9 = fill(x5, 1, x8)
    x10 = objects(x9, T, T, T)
    x11 = first(x10)
    x12 = subgrid(x11, x9)
    x13 = shape(x12)
    x14 = subtract(x13, DOWN)
    x15 = crop(x12, DOWN, x14)
    x16 = vconcat(x15, x15)
    x17 = vconcat(x16, x16)
    x18 = vconcat(x17, x17)
    x19 = hmirror(x18)
    x20 = crop(x19, ORIGIN, x1)
    O = hmirror(x20)
    return O


def solve_8d510a79(I):
    x1 = ofcolor(I, 1)
    x2 = ofcolor(I, 2)
    x3 = ofcolor(I, 5)
    x4 = uppermost(x3)
    x5 = chain(toivec, decrement, double)
    x6 = lbind(greater, x4)
    x7 = compose(x6, first)
    x8 = chain(invert, x5, x7)
    x9 = fork(shoot, identity, x8)
    x10 = compose(x5, x7)
    x11 = fork(shoot, identity, x10)
    x12 = lbind(matcher, x7)
    x13 = compose(x12, x7)
    x14 = fork(sfilter, x11, x13)
    x15 = mapply(x9, x1)
    x16 = mapply(x14, x2)
    x17 = underfill(I, 2, x16)
    O = fill(x17, 1, x15)
    return O


def solve_cdecee7f(I):
    x1 = objects(I, T, F, T)
    x2 = astuple(1, 3)
    x3 = size(x1)
    x4 = order(x1, leftmost)
    x5 = apply(color, x4)
    x6 = rbind(canvas, UNITY)
    x7 = apply(x6, x5)
    x8 = merge(x7)
    x9 = dmirror(x8)
    x10 = subtract(9, x3)
    x11 = astuple(1, x10)
    x12 = canvas(0, x11)
    x13 = hconcat(x9, x12)
    x14 = hsplit(x13, 3)
    x15 = merge(x14)
    x16 = crop(x15, ORIGIN, x2)
    x17 = crop(x15, DOWN, x2)
    x18 = crop(x15, G2x0, x2)
    x19 = vmirror(x17)
    x20 = vconcat(x16, x19)
    O = vconcat(x20, x18)
    return O


def solve_3345333e(I):
    x1 = leastcolor(I)
    x2 = ofcolor(I, x1)
    x3 = cover(I, x2)
    x4 = leastcolor(x3)
    x5 = ofcolor(x3, x4)
    x6 = neighbors(ORIGIN)
    x7 = mapply(neighbors, x6)
    x8 = vmirror(x5)
    x9 = lbind(shift, x8)
    x10 = apply(x9, x7)
    x11 = rbind(intersection, x5)
    x12 = compose(size, x11)
    x13 = argmax(x10, x12)
    O = fill(x3, x4, x13)
    return O


def solve_b190f7f5(I):
    x1 = portrait(I)
    x2 = branch(x1, vsplit, hsplit)
    x3 = x2(I, 2)
    x4 = argmin(x3, numcolors)
    x5 = argmax(x3, numcolors)
    x6 = width(x5)
    x7 = rbind(repeat, x6)
    x8 = chain(dmirror, merge, x7)
    x9 = upscale(x5, x6)
    x10 = x8(x4)
    x11 = x8(x10)
    x12 = ofcolor(x11, 0)
    O = fill(x9, 0, x12)
    return O


def solve_caa06a1f(I):
    x1 = asobject(I)
    x2 = shape(I)
    x3 = decrement(x2)
    x4 = index(I, x3)
    x5 = double(x2)
    x6 = canvas(x4, x5)
    x7 = paint(x6, x1)
    x8 = objects(x7, F, F, T)
    x9 = first(x8)
    x10 = shift(x9, LEFT)
    x11 = vperiod(x10)
    x12 = hperiod(x10)
    x13 = neighbors(ORIGIN)
    x14 = lbind(mapply, neighbors)
    x15 = power(x14, 2)
    x16 = x15(x13)
    x17 = astuple(x11, x12)
    x18 = lbind(multiply, x17)
    x19 = apply(x18, x16)
    x20 = lbind(shift, x10)
    x21 = mapply(x20, x19)
    O = paint(I, x21)
    return O


def solve_e21d9049(I):
    x1 = asindices(I)
    x2 = leastcolor(I)
    x3 = objects(I, T, F, T)
    x4 = ofcolor(I, x2)
    x5 = merge(x3)
    x6 = shape(x5)
    x7 = neighbors(ORIGIN)
    x8 = lbind(mapply, neighbors)
    x9 = power(x8, 2)
    x10 = x9(x7)
    x11 = lbind(multiply, x6)
    x12 = lbind(shift, x5)
    x13 = apply(x11, x10)
    x14 = mapply(x12, x13)
    x15 = lbind(hmatching, x4)
    x16 = lbind(vmatching, x4)
    x17 = fork(either, x15, x16)
    x18 = compose(x17, initset)
    x19 = paint(I, x14)
    x20 = sfilter(x1, x18)
    x21 = difference(x1, x20)
    O = cover(x19, x21)
    return O


def solve_d89b689b(I):
    x1 = objects(I, T, F, T)
    x2 = ofcolor(I, 8)
    x3 = sizefilter(x1, 1)
    x4 = apply(initset, x2)
    x5 = lbind(argmin, x4)
    x6 = lbind(rbind, manhattan)
    x7 = compose(x5, x6)
    x8 = fork(recolor, color, x7)
    x9 = mapply(x8, x3)
    x10 = merge(x3)
    x11 = cover(I, x10)
    O = paint(x11, x9)
    return O


def solve_746b3537(I):
    x1 = chain(size, dedupe, first)
    x2 = x1(I)
    x3 = equality(x2, 1)
    x4 = branch(x3, dmirror, identity)
    x5 = x4(I)
    x6 = objects(x5, T, F, F)
    x7 = order(x6, leftmost)
    x8 = apply(color, x7)
    x9 = repeat(x8, 1)
    O = x4(x9)
    return O


def solve_63613498(I):
    x1 = crop(I, ORIGIN, G3x3)
    x2 = ofcolor(x1, 0)
    x3 = asindices(x1)
    x4 = difference(x3, x2)
    x5 = normalize(x4)
    x6 = objects(I, T, F, T)
    x7 = compose(toindices, normalize)
    x8 = matcher(x7, x5)
    x9 = mfilter(x6, x8)
    x10 = fill(I, 5, x9)
    x11 = asobject(x1)
    O = paint(x10, x11)
    return O


def solve_06df4c85(I):
    x1 = partition(I)
    x2 = mostcolor(I)
    x3 = ofcolor(I, x2)
    x4 = colorfilter(x1, 0)
    x5 = argmax(x1, size)
    x6 = difference(x1, x4)
    x7 = remove(x5, x6)
    x8 = merge(x7)
    x9 = product(x8, x8)
    x10 = power(first, 2)
    x11 = compose(first, last)
    x12 = fork(equality, x10, x11)
    x13 = sfilter(x9, x12)
    x14 = compose(last, first)
    x15 = power(last, 2)
    x16 = fork(connect, x14, x15)
    x17 = fork(recolor, color, x16)
    x18 = apply(x17, x13)
    x19 = fork(either, vline, hline)
    x20 = mfilter(x18, x19)
    x21 = paint(I, x20)
    O = fill(x21, x2, x3)
    return O


def solve_f9012d9b(I):
    x1 = objects(I, T, F, F)
    x2 = ofcolor(I, 0)
    x3 = lbind(contained, 0)
    x4 = chain(flip, x3, palette)
    x5 = mfilter(x1, x4)
    x6 = vsplit(I, 2)
    x7 = hsplit(I, 2)
    x8 = extract(x6, x4)
    x9 = extract(x7, x4)
    x10 = asobject(x8)
    x11 = asobject(x9)
    x12 = vperiod(x10)
    x13 = hperiod(x11)
    x14 = neighbors(ORIGIN)
    x15 = mapply(neighbors, x14)
    x16 = astuple(x12, x13)
    x17 = rbind(multiply, x16)
    x18 = apply(x17, x15)
    x19 = lbind(shift, x5)
    x20 = mapply(x19, x18)
    x21 = paint(I, x20)
    O = subgrid(x2, x21)
    return O


def solve_4522001f(I):
    x1 = objects(I, F, F, T)
    x2 = first(x1)
    x3 = toindices(x2)
    x4 = contained(G0x2, x3)
    x5 = contained(G2x2, x3)
    x6 = contained(G2x0, x3)
    x7 = astuple(9, 9)
    x8 = canvas(0, x7)
    x9 = astuple(3, ORIGIN)
    x10 = initset(x9)
    x11 = upscale(x10, 2)
    x12 = upscale(x11, 2)
    x13 = shape(x12)
    x14 = shift(x12, x13)
    x15 = combine(x12, x14)
    x16 = paint(x8, x15)
    x17 = rot90(x16)
    x18 = rot180(x16)
    x19 = rot270(x16)
    x20 = branch(x4, x17, x16)
    x21 = branch(x5, x18, x20)
    O = branch(x6, x19, x21)
    return O


def solve_a48eeaf7(I):
    x1 = ofcolor(I, 2)
    x2 = outbox(x1)
    x3 = apply(initset, x2)
    x4 = ofcolor(I, 5)
    x5 = lbind(argmin, x3)
    x6 = lbind(lbind, manhattan)
    x7 = compose(x6, initset)
    x8 = compose(x5, x7)
    x9 = mapply(x8, x4)
    x10 = cover(I, x4)
    O = fill(x10, 5, x9)
    return O


def solve_eb5a1d5d(I):
    x1 = compose(dmirror, dedupe)
    x2 = x1(I)
    x3 = x1(x2)
    x4 = fork(remove, last, identity)
    x5 = compose(hmirror, x4)
    x6 = fork(vconcat, identity, x5)
    x7 = x6(x3)
    x8 = dmirror(x7)
    O = x6(x8)
    return O


def solve_e179c5f4(I):
    x1 = height(I)
    x2 = ofcolor(I, 1)
    x3 = first(x2)
    x4 = shoot(x3, UP_RIGHT)
    x5 = fill(I, 1, x4)
    x6 = ofcolor(x5, 1)
    x7 = urcorner(x6)
    x8 = shoot(x7, NEG_UNITY)
    x9 = fill(x5, 1, x8)
    x10 = ofcolor(x9, 1)
    x11 = subgrid(x10, x9)
    x12 = height(x11)
    x13 = width(x11)
    x14 = decrement(x12)
    x15 = astuple(x14, x13)
    x16 = ulcorner(x10)
    x17 = crop(x9, x16, x15)
    x18 = repeat(x17, 9)
    x19 = merge(x18)
    x20 = astuple(x1, x13)
    x21 = crop(x19, ORIGIN, x20)
    x22 = hmirror(x21)
    O = replace(x22, 0, 8)
    return O


def solve_228f6490(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, 0)
    x3 = rbind(bordering, I)
    x4 = compose(flip, x3)
    x5 = sfilter(x2, x4)
    x6 = first(x5)
    x7 = last(x5)
    x8 = difference(x1, x2)
    x9 = compose(normalize, toindices)
    x10 = x9(x6)
    x11 = x9(x7)
    x12 = matcher(x9, x10)
    x13 = matcher(x9, x11)
    x14 = extract(x8, x12)
    x15 = extract(x8, x13)
    x16 = ulcorner(x6)
    x17 = ulcorner(x7)
    x18 = ulcorner(x14)
    x19 = ulcorner(x15)
    x20 = subtract(x16, x18)
    x21 = subtract(x17, x19)
    x22 = move(I, x14, x20)
    O = move(x22, x15, x21)
    return O


def solve_995c5fa3(I):
    x1 = hsplit(I, 3)
    x2 = astuple(2, 1)
    x3 = rbind(ofcolor, 0)
    x4 = compose(ulcorner, x3)
    x5 = compose(size, x3)
    x6 = matcher(x5, 0)
    x7 = matcher(x4, UNITY)
    x8 = matcher(x4, DOWN)
    x9 = matcher(x4, x2)
    x10 = rbind(multiply, 3)
    x11 = power(double, 2)
    x12 = compose(double, x6)
    x13 = chain(x11, double, x7)
    x14 = compose(x10, x8)
    x15 = compose(x11, x9)
    x16 = fork(add, x12, x13)
    x17 = fork(add, x14, x15)
    x18 = fork(add, x16, x17)
    x19 = rbind(canvas, UNITY)
    x20 = compose(x19, x18)
    x21 = apply(x20, x1)
    x22 = merge(x21)
    O = hupscale(x22, 3)
    return O


def solve_d06dbe63(I):
    x1 = ofcolor(I, 8)
    x2 = center(x1)
    x3 = connect(ORIGIN, DOWN)
    x4 = connect(ORIGIN, G0x2)
    x5 = combine(x3, x4)
    x6 = subtract(x2, G2x0)
    x7 = shift(x5, x6)
    x8 = astuple(-2, 2)
    x9 = interval(0, 5, 1)
    x10 = lbind(multiply, x8)
    x11 = apply(x10, x9)
    x12 = lbind(shift, x7)
    x13 = mapply(x12, x11)
    x14 = fill(I, 5, x13)
    x15 = rot180(x14)
    x16 = ofcolor(x15, 8)
    x17 = center(x16)
    x18 = subtract(x17, x6)
    x19 = shift(x13, x18)
    x20 = toivec(-2)
    x21 = shift(x19, x20)
    x22 = fill(x15, 5, x21)
    O = rot180(x22)
    return O


def solve_36fdfd69(I):
    x1 = upscale(I, 2)
    x2 = objects(x1, T, T, T)
    x3 = colorfilter(x2, 2)
    x4 = fork(manhattan, first, last)
    x5 = lbind(greater, 5)
    x6 = compose(x5, x4)
    x7 = product(x3, x3)
    x8 = sfilter(x7, x6)
    x9 = apply(merge, x8)
    x10 = mapply(delta, x9)
    x11 = fill(x1, 4, x10)
    x12 = merge(x3)
    x13 = paint(x11, x12)
    O = downscale(x13, 2)
    return O


def solve_0a938d79(I):
    x1 = portrait(I)
    x2 = branch(x1, dmirror, identity)
    x3 = x2(I)
    x4 = fgpartition(x3)
    x5 = merge(x4)
    x6 = chain(double, decrement, width)
    x7 = x6(x5)
    x8 = compose(vfrontier, tojvec)
    x9 = lbind(mapply, x8)
    x10 = rbind(interval, x7)
    x11 = width(x3)
    x12 = rbind(x10, x11)
    x13 = chain(x9, x12, leftmost)
    x14 = fork(recolor, color, x13)
    x15 = mapply(x14, x4)
    x16 = paint(x3, x15)
    O = x2(x16)
    return O


def solve_045e512c(I):
    x1 = objects(I, T, T, T)
    x2 = argmax(x1, size)
    x3 = remove(x2, x1)
    x4 = lbind(shift, x2)
    x5 = lbind(mapply, x4)
    x6 = double(10)
    x7 = interval(4, x6, 4)
    x8 = rbind(apply, x7)
    x9 = lbind(position, x2)
    x10 = lbind(rbind, multiply)
    x11 = chain(x8, x10, x9)
    x12 = compose(x5, x11)
    x13 = fork(recolor, color, x12)
    x14 = mapply(x13, x3)
    O = paint(I, x14)
    return O


def solve_82819916(I):
    x1 = objects(I, F, T, T)
    x2 = argmax(x1, size)
    x3 = remove(x2, x1)
    x4 = normalize(x2)
    x5 = compose(last, last)
    x6 = rbind(argmin, x5)
    x7 = compose(first, x6)
    x8 = fork(other, palette, x7)
    x9 = x7(x4)
    x10 = matcher(first, x9)
    x11 = sfilter(x4, x10)
    x12 = difference(x4, x11)
    x13 = compose(toivec, uppermost)
    x14 = lbind(shift, x11)
    x15 = lbind(shift, x12)
    x16 = compose(x14, x13)
    x17 = compose(x15, x13)
    x18 = fork(recolor, x7, x16)
    x19 = fork(recolor, x8, x17)
    x20 = fork(combine, x18, x19)
    x21 = mapply(x20, x3)
    O = paint(I, x21)
    return O


def solve_99fa7670(I):
    x1 = shape(I)
    x2 = objects(I, T, F, T)
    x3 = rbind(shoot, RIGHT)
    x4 = compose(x3, center)
    x5 = fork(recolor, color, x4)
    x6 = mapply(x5, x2)
    x7 = paint(I, x6)
    x8 = add(x1, DOWN_LEFT)
    x9 = initset(x8)
    x10 = recolor(0, x9)
    x11 = objects(x7, T, F, T)
    x12 = insert(x10, x11)
    x13 = order(x12, uppermost)
    x14 = first(x13)
    x15 = remove(x10, x13)
    x16 = remove(x14, x13)
    x17 = compose(lrcorner, first)
    x18 = compose(lrcorner, last)
    x19 = fork(connect, x17, x18)
    x20 = compose(color, first)
    x21 = fork(recolor, x20, x19)
    x22 = pair(x15, x16)
    x23 = mapply(x21, x22)
    O = underpaint(x7, x23)
    return O


def solve_72322fa7(I):
    x1 = objects(I, F, T, T)
    x2 = matcher(numcolors, 1)
    x3 = sfilter(x1, x2)
    x4 = difference(x1, x3)
    x5 = lbind(matcher, first)
    x6 = compose(x5, mostcolor)
    x7 = fork(sfilter, identity, x6)
    x8 = fork(difference, identity, x7)
    x9 = lbind(occurrences, I)
    x10 = compose(x9, x7)
    x11 = compose(x9, x8)
    x12 = compose(ulcorner, x8)
    x13 = fork(subtract, ulcorner, x12)
    x14 = lbind(rbind, add)
    x15 = compose(x14, x13)
    x16 = fork(apply, x15, x11)
    x17 = lbind(lbind, shift)
    x18 = compose(x17, normalize)
    x19 = fork(mapply, x18, x10)
    x20 = fork(mapply, x18, x16)
    x21 = mapply(x19, x4)
    x22 = mapply(x20, x4)
    x23 = paint(I, x21)
    O = paint(x23, x22)
    return O


def solve_855e0971(I):
    x1 = rot90(I)
    x2 = frontiers(I)
    x3 = sfilter(x2, hline)
    x4 = size(x3)
    x6 = positive(x4)
    x7 = branch(x6, identity, dmirror)
    x8 = x7(I)
    x9 = rbind(subgrid, x8)
    x10 = matcher(color, 0)
    x11 = compose(flip, x10)
    x12 = partition(x8)
    x13 = sfilter(x12, x11)
    x14 = rbind(ofcolor, 0)
    x15 = lbind(mapply, vfrontier)
    x16 = chain(x15, x14, x9)
    x17 = fork(shift, x16, ulcorner)
    x18 = fork(intersection, toindices, x17)
    x19 = mapply(x18, x13)
    x20 = fill(x8, 0, x19)
    O = x7(x20)
    return O


def solve_a78176bb(I):
    x1 = palette(I)
    x2 = objects(I, T, F, T)
    x3 = remove(0, x1)
    x4 = other(x3, 5)
    x5 = colorfilter(x2, 5)
    x6 = lbind(index, I)
    x7 = compose(x6, urcorner)
    x8 = matcher(x7, 5)
    x9 = sfilter(x5, x8)
    x10 = difference(x5, x9)
    x11 = apply(urcorner, x9)
    x12 = apply(llcorner, x10)
    x13 = rbind(add, UP_RIGHT)
    x14 = rbind(add, DOWN_LEFT)
    x15 = apply(x13, x11)
    x16 = apply(x14, x12)
    x17 = rbind(shoot, UNITY)
    x18 = rbind(shoot, NEG_UNITY)
    x19 = fork(combine, x17, x18)
    x20 = mapply(x19, x15)
    x21 = mapply(x19, x16)
    x22 = combine(x20, x21)
    x23 = fill(I, x4, x22)
    O = replace(x23, 5, 0)
    return O


def solve_952a094c(I):
    x1 = objects(I, T, F, T)
    x2 = sizefilter(x1, 1)
    x3 = argmax(x1, size)
    x4 = outbox(x3)
    x5 = corners(x4)
    x6 = lbind(rbind, manhattan)
    x7 = lbind(argmax, x2)
    x8 = chain(x7, x6, initset)
    x9 = compose(color, x8)
    x10 = fork(astuple, x9, identity)
    x11 = apply(x10, x5)
    x12 = merge(x2)
    x13 = cover(I, x12)
    O = paint(x13, x11)
    return O


def solve_6d58a25d(I):
    x1 = objects(I, T, T, T)
    x2 = argmax(x1, size)
    x3 = remove(x2, x1)
    x4 = merge(x3)
    x5 = color(x4)
    x6 = uppermost(x2)
    x7 = rbind(greater, x6)
    x8 = compose(x7, uppermost)
    x9 = rbind(vmatching, x2)
    x10 = fork(both, x9, x8)
    x11 = sfilter(x3, x10)
    x12 = increment(x6)
    x13 = rbind(greater, x12)
    x14 = compose(x13, first)
    x15 = rbind(sfilter, x14)
    x16 = chain(x15, vfrontier, center)
    x17 = mapply(x16, x11)
    O = underfill(I, x5, x17)
    return O


def solve_6aa20dc0(I):
    x1 = objects(I, F, T, T)
    x2 = argmax(x1, numcolors)
    x3 = normalize(x2)
    x4 = lbind(matcher, first)
    x5 = compose(x4, mostcolor)
    x6 = fork(sfilter, identity, x5)
    x7 = fork(difference, identity, x6)
    x8 = lbind(rbind, upscale)
    x9 = interval(1, 4, 1)
    x10 = apply(x8, x9)
    x11 = initset(identity)
    x12 = insert(vmirror, x11)
    x13 = insert(hmirror, x12)
    x14 = insert(cmirror, x13)
    x15 = insert(dmirror, x14)
    x16 = fork(compose, first, last)
    x17 = lbind(occurrences, I)
    x18 = lbind(lbind, shift)
    x19 = compose(x17, x7)
    x20 = product(x15, x10)
    x21 = apply(x16, x20)
    x22 = rapply(x21, x3)
    x23 = fork(mapply, x18, x19)
    x24 = mapply(x23, x22)
    O = paint(I, x24)
    return O


def solve_e6721834(I):
    x1 = portrait(I)
    x2 = branch(x1, vsplit, hsplit)
    x3 = x2(I, 2)
    x4 = order(x3, numcolors)
    x5 = first(x4)
    x6 = last(x4)
    x7 = objects(x6, F, F, T)
    x8 = merge(x7)
    x9 = mostcolor(x8)
    x10 = matcher(first, x9)
    x11 = compose(flip, x10)
    x12 = rbind(sfilter, x11)
    x13 = lbind(occurrences, x5)
    x14 = compose(x13, x12)
    x15 = chain(positive, size, x14)
    x16 = sfilter(x7, x15)
    x17 = chain(first, x13, x12)
    x18 = compose(ulcorner, x12)
    x19 = fork(subtract, x17, x18)
    x20 = fork(shift, identity, x19)
    x21 = apply(x20, x16)
    x22 = compose(decrement, width)
    x23 = chain(positive, decrement, x22)
    x24 = mfilter(x21, x23)
    O = paint(x5, x24)
    return O


def solve_447fd412(I):
    x1 = objects(I, F, T, T)
    x2 = argmax(x1, numcolors)
    x3 = normalize(x2)
    x4 = lbind(matcher, first)
    x5 = compose(x4, mostcolor)
    x6 = fork(sfilter, identity, x5)
    x7 = fork(difference, identity, x6)
    x8 = lbind(rbind, upscale)
    x9 = interval(1, 4, 1)
    x10 = apply(x8, x9)
    x11 = lbind(recolor, 0)
    x12 = compose(x11, outbox)
    x13 = fork(combine, identity, x12)
    x14 = lbind(occurrences, I)
    x15 = lbind(rbind, subtract)
    x16 = lbind(apply, increment)
    x17 = lbind(lbind, shift)
    x18 = chain(x15, ulcorner, x7)
    x19 = chain(x14, x13, x7)
    x20 = fork(apply, x18, x19)
    x21 = compose(x16, x20)
    x22 = fork(mapply, x17, x21)
    x23 = rapply(x10, x3)
    x24 = mapply(x22, x23)
    O = paint(I, x24)
    return O


def solve_2bcee788(I):
    x1 = mostcolor(I)
    x2 = objects(I, T, F, T)
    x3 = replace(I, x1, 3)
    x4 = argmax(x2, size)
    x5 = argmin(x2, size)
    x6 = position(x4, x5)
    x7 = first(x6)
    x8 = last(x6)
    x9 = subgrid(x4, x3)
    x10 = hline(x5)
    x11 = hmirror(x9)
    x12 = vmirror(x9)
    x13 = branch(x10, x11, x12)
    x14 = branch(x10, x7, 0)
    x15 = branch(x10, 0, x8)
    x16 = asobject(x13)
    x17 = matcher(first, 3)
    x18 = compose(flip, x17)
    x19 = sfilter(x16, x18)
    x20 = ulcorner(x4)
    x21 = shape(x4)
    x22 = astuple(x14, x15)
    x23 = multiply(x21, x22)
    x24 = add(x20, x23)
    x25 = shift(x19, x24)
    O = paint(x3, x25)
    return O


def solve_776ffc46(I):
    x1 = objects(I, T, F, T)
    x2 = colorfilter(x1, 5)
    x3 = fork(equality, toindices, box)
    x4 = extract(x2, x3)
    x5 = inbox(x4)
    x6 = subgrid(x5, I)
    x7 = asobject(x6)
    x8 = matcher(first, 0)
    x9 = compose(flip, x8)
    x10 = sfilter(x7, x9)
    x11 = normalize(x10)
    x12 = toindices(x11)
    x13 = compose(toindices, normalize)
    x14 = matcher(x13, x12)
    x15 = mfilter(x1, x14)
    x16 = color(x11)
    O = fill(I, x16, x15)
    return O


def solve_f35d900a(I):
    x1 = objects(I, T, F, T)
    x2 = palette(I)
    x3 = remove(0, x2)
    x4 = lbind(other, x3)
    x5 = compose(x4, color)
    x6 = fork(recolor, x5, outbox)
    x7 = mapply(x6, x1)
    x8 = mapply(toindices, x1)
    x9 = box(x8)
    x10 = difference(x9, x8)
    x11 = lbind(argmin, x8)
    x12 = rbind(compose, initset)
    x13 = lbind(rbind, manhattan)
    x14 = chain(x12, x13, initset)
    x15 = chain(initset, x11, x14)
    x16 = fork(manhattan, initset, x15)
    x17 = compose(even, x16)
    x18 = sfilter(x10, x17)
    x19 = paint(I, x7)
    O = fill(x19, 5, x18)
    return O


def solve_0dfd9992(I):
    x1 = height(I)
    x2 = width(I)
    x3 = partition(I)
    x4 = colorfilter(x3, 0)
    x5 = difference(x3, x4)
    x6 = merge(x5)
    x7 = astuple(x1, 1)
    x8 = astuple(1, x2)
    x9 = decrement(x1)
    x10 = decrement(x2)
    x11 = toivec(x10)
    x12 = tojvec(x9)
    x13 = crop(I, x11, x8)
    x14 = crop(I, x12, x7)
    x15 = asobject(x14)
    x16 = asobject(x13)
    x17 = vperiod(x15)
    x18 = hperiod(x16)
    x19 = astuple(x17, x18)
    x20 = lbind(multiply, x19)
    x21 = neighbors(ORIGIN)
    x22 = mapply(neighbors, x21)
    x23 = apply(x20, x22)
    x24 = lbind(shift, x6)
    x25 = mapply(x24, x23)
    O = paint(I, x25)
    return O


def solve_29ec7d0e(I):
    x1 = height(I)
    x2 = width(I)
    x3 = partition(I)
    x4 = colorfilter(x3, 0)
    x5 = difference(x3, x4)
    x6 = merge(x5)
    x7 = astuple(x1, 1)
    x8 = astuple(1, x2)
    x9 = decrement(x1)
    x10 = decrement(x2)
    x11 = toivec(x10)
    x12 = tojvec(x9)
    x13 = crop(I, x11, x8)
    x14 = crop(I, x12, x7)
    x15 = asobject(x14)
    x16 = asobject(x13)
    x17 = vperiod(x15)
    x18 = hperiod(x16)
    x19 = astuple(x17, x18)
    x20 = lbind(multiply, x19)
    x21 = neighbors(ORIGIN)
    x22 = mapply(neighbors, x21)
    x23 = apply(x20, x22)
    x24 = lbind(shift, x6)
    x25 = mapply(x24, x23)
    O = paint(I, x25)
    return O


def solve_36d67576(I):
    x1 = objects(I, F, F, T)
    x2 = argmax(x1, numcolors)
    x3 = astuple(2, 4)
    x4 = rbind(contained, x3)
    x5 = compose(x4, first)
    x6 = rbind(sfilter, x5)
    x7 = lbind(rbind, subtract)
    x8 = lbind(occurrences, I)
    x9 = lbind(lbind, shift)
    x10 = compose(x7, ulcorner)
    x11 = chain(x10, x6, normalize)
    x12 = chain(x8, x6, normalize)
    x13 = fork(apply, x11, x12)
    x14 = compose(x9, normalize)
    x15 = fork(mapply, x14, x13)
    x16 = astuple(cmirror, dmirror)
    x17 = astuple(hmirror, vmirror)
    x18 = combine(x16, x17)
    x19 = product(x18, x18)
    x20 = fork(compose, first, last)
    x21 = apply(x20, x19)
    x22 = totuple(x21)
    x23 = combine(x18, x22)
    x24 = rapply(x23, x2)
    x25 = mapply(x15, x24)
    O = paint(I, x25)
    return O


def solve_98cf29f8(I):
    x1 = fgpartition(I)
    x2 = fork(multiply, height, width)
    x3 = fork(equality, size, x2)
    x4 = extract(x1, x3)
    x5 = other(x1, x4)
    x6 = color(x5)
    x7 = rbind(greater, 3)
    x8 = rbind(toobject, I)
    x9 = rbind(colorcount, x6)
    x10 = chain(x8, ineighbors, last)
    x11 = chain(x7, x9, x10)
    x12 = sfilter(x5, x11)
    x13 = outbox(x12)
    x14 = backdrop(x13)
    x15 = cover(I, x5)
    x16 = gravitate(x14, x4)
    x17 = shift(x14, x16)
    O = fill(x15, x6, x17)
    return O


def solve_469497ad(I):
    x1 = numcolors(I)
    x2 = decrement(x1)
    x3 = upscale(I, x2)
    x4 = objects(x3, F, F, T)
    x5 = argmin(x4, size)
    x6 = ulcorner(x5)
    x7 = llcorner(x5)
    x8 = shoot(x6, NEG_UNITY)
    x9 = shoot(x6, UNITY)
    x10 = shoot(x7, DOWN_LEFT)
    x11 = shoot(x7, UP_RIGHT)
    x12 = combine(x8, x9)
    x13 = combine(x10, x11)
    x14 = combine(x12, x13)
    x15 = underfill(x3, 2, x14)
    x16 = objects(x15, T, F, T)
    x17 = argmax(x16, lrcorner)
    O = paint(x15, x17)
    return O


def solve_39e1d7f9(I):
    x1 = fgpartition(I)
    x2 = objects(I, T, F, T)
    x3 = order(x1, height)
    x4 = last(x3)
    x5 = remove(x4, x3)
    x6 = last(x5)
    x7 = color(x6)
    x8 = colorfilter(x2, x7)
    x9 = power(outbox, 2)
    x10 = rbind(toobject, I)
    x11 = mostcolor(I)
    x12 = lbind(remove, x11)
    x13 = chain(size, x12, palette)
    x14 = chain(x13, x10, x9)
    x15 = argmax(x8, x14)
    x16 = ulcorner(x15)
    x17 = shape(x15)
    x18 = subtract(x16, x17)
    x19 = decrement(x18)
    x20 = multiply(x17, 3)
    x21 = add(x20, G2x2)
    x22 = crop(I, x19, x21)
    x23 = asobject(x22)
    x24 = apply(ulcorner, x8)
    x25 = increment(x17)
    x26 = rbind(subtract, x25)
    x27 = apply(x26, x24)
    x28 = lbind(shift, x23)
    x29 = mapply(x28, x27)
    O = paint(I, x29)
    return O


def solve_484b58aa(I):
    x1 = height(I)
    x2 = width(I)
    x3 = partition(I)
    x4 = colorfilter(x3, 0)
    x5 = difference(x3, x4)
    x6 = merge(x5)
    x7 = astuple(x1, 2)
    x8 = astuple(2, x2)
    x9 = power(decrement, 2)
    x10 = x9(x1)
    x11 = x9(x2)
    x12 = toivec(x11)
    x13 = tojvec(x10)
    x14 = crop(I, x12, x8)
    x15 = crop(I, x13, x7)
    x16 = asobject(x15)
    x17 = asobject(x14)
    x18 = vperiod(x16)
    x19 = hperiod(x17)
    x20 = astuple(x18, x19)
    x21 = lbind(multiply, x20)
    x22 = neighbors(ORIGIN)
    x23 = mapply(neighbors, x22)
    x24 = apply(x21, x23)
    x25 = lbind(shift, x6)
    x26 = mapply(x25, x24)
    O = paint(I, x26)
    return O


def solve_3befdf3e(I):
    x1 = objects(I, F, F, T)
    x2 = leastcolor(I)
    x3 = palette(I)
    x4 = remove(0, x3)
    x5 = other(x4, x2)
    x6 = switch(I, x2, x5)
    x7 = compose(width, inbox)
    x8 = lbind(power, outbox)
    x9 = compose(x8, x7)
    x10 = initset(x9)
    x11 = lbind(rapply, x10)
    x12 = chain(initset, first, x11)
    x13 = fork(rapply, x12, identity)
    x14 = compose(first, x13)
    x15 = compose(backdrop, x14)
    x16 = lbind(chain, backdrop)
    x17 = lbind(x16, inbox)
    x18 = compose(x17, x9)
    x19 = lbind(apply, initset)
    x20 = chain(x19, corners, x15)
    x21 = fork(mapply, x18, x20)
    x22 = fork(intersection, x15, x21)
    x23 = mapply(x15, x1)
    x24 = mapply(x22, x1)
    x25 = underfill(x6, x5, x23)
    O = fill(x25, 0, x24)
    return O


def solve_9aec4887(I):
    x1 = objects(I, F, T, T)
    x2 = argmin(x1, numcolors)
    x3 = other(x1, x2)
    x4 = subgrid(x3, I)
    x5 = normalize(x2)
    x6 = shift(x5, UNITY)
    x7 = toindices(x6)
    x8 = normalize(x3)
    x9 = lbind(argmin, x8)
    x11 = lbind(rbind, manhattan)
    x12 = rbind(compose, initset)
    x13 = chain(x12, x11, initset)
    x14 = chain(first, x9, x13)
    x15 = fork(astuple, x14, identity)
    x16 = apply(x15, x7)
    x17 = paint(x4, x16)
    x18 = fork(connect, ulcorner, lrcorner)
    x19 = x18(x7)
    x20 = fork(combine, identity, vmirror)
    x21 = x20(x19)
    x22 = intersection(x7, x21)
    O = fill(x17, 8, x22)
    return O


def solve_49d1d64f(I):
    x1 = shape(I)
    x2 = add(x1, 2)
    x3 = canvas(0, x2)
    x4 = asobject(I)
    x5 = shift(x4, UNITY)
    x6 = paint(x3, x5)
    x7 = asindices(x3)
    x8 = fork(difference, box, corners)
    x9 = x8(x7)
    x10 = lbind(lbind, manhattan)
    x11 = rbind(compose, initset)
    x12 = chain(x11, x10, initset)
    x13 = lbind(argmin, x5)
    x14 = chain(first, x13, x12)
    x15 = fork(astuple, x14, identity)
    x16 = apply(x15, x9)
    O = paint(x6, x16)
    return O


def solve_57aa92db(I):
    x1 = objects(I, F, T, T)
    x2 = objects(I, T, F, T)
    x3 = lbind(lbind, colorcount)
    x4 = fork(apply, x3, palette)
    x5 = compose(maximum, x4)
    x6 = compose(minimum, x4)
    x7 = fork(subtract, x5, x6)
    x8 = argmax(x1, x7)
    x9 = leastcolor(x8)
    x10 = normalize(x8)
    x11 = matcher(first, x9)
    x12 = sfilter(x10, x11)
    x13 = ulcorner(x12)
    x14 = colorfilter(x2, x9)
    x15 = rbind(toobject, I)
    x16 = lbind(remove, 0)
    x17 = chain(first, x16, palette)
    x18 = chain(x17, x15, outbox)
    x19 = lbind(multiply, x13)
    x20 = compose(x19, width)
    x21 = fork(subtract, ulcorner, x20)
    x22 = lbind(shift, x10)
    x23 = compose(x22, x21)
    x24 = fork(upscale, x23, width)
    x25 = fork(recolor, x18, x24)
    x26 = mapply(x25, x14)
    x27 = paint(I, x26)
    x28 = merge(x2)
    O = paint(x27, x28)
    return O


def solve_aba27056(I):
    x1 = objects(I, T, F, T)
    x2 = mapply(toindices, x1)
    x3 = box(x2)
    x4 = difference(x3, x2)
    x5 = delta(x2)
    x6 = position(x5, x4)
    x7 = interval(0, 9, 1)
    x8 = lbind(multiply, x6)
    x9 = apply(x8, x7)
    x10 = lbind(shift, x4)
    x11 = mapply(x10, x9)
    x12 = fill(I, 4, x5)
    x13 = fill(x12, 4, x11)
    x14 = corners(x4)
    x15 = ofcolor(x13, 0)
    x16 = rbind(toobject, x13)
    x17 = rbind(colorcount, 0)
    x18 = chain(x17, x16, dneighbors)
    x19 = matcher(x18, 2)
    x20 = rbind(adjacent, x2)
    x21 = rbind(adjacent, x11)
    x22 = fork(both, x20, x21)
    x23 = compose(x22, initset)
    x24 = sfilter(x15, x19)
    x25 = sfilter(x24, x23)
    x26 = product(x14, x25)
    x27 = fork(subtract, last, first)
    x28 = fork(shoot, first, x27)
    x29 = mapply(x28, x26)
    O = fill(x13, 4, x29)
    return O


def solve_f1cefba8(I):
    x1 = palette(I)
    x2 = objects(I, F, F, T)
    x3 = ofcolor(I, 0)
    x4 = first(x2)
    x5 = ulcorner(x4)
    x6 = subgrid(x4, I)
    x7 = power(trim, 2)
    x8 = x7(x6)
    x9 = asindices(x8)
    x10 = shift(x9, G2x2)
    x11 = fill(x6, 0, x10)
    x12 = leastcolor(x11)
    x13 = remove(0, x1)
    x14 = other(x13, x12)
    x15 = ofcolor(x11, x12)
    x16 = shift(x15, x5)
    x17 = ofcolor(I, x12)
    x18 = uppermost(x17)
    x19 = lowermost(x17)
    x20 = matcher(first, x18)
    x21 = matcher(first, x19)
    x22 = fork(either, x20, x21)
    x23 = sfilter(x16, x22)
    x24 = difference(x16, x23)
    x25 = mapply(vfrontier, x23)
    x26 = mapply(hfrontier, x24)
    x27 = combine(x25, x26)
    x28 = intersection(x3, x27)
    x29 = fill(I, x14, x27)
    O = fill(x29, x12, x28)
    return O


def solve_1e32b0e9(I):
    x1 = height(I)
    x2 = mostcolor(I)
    x3 = asobject(I)
    x4 = subtract(x1, 2)
    x5 = divide(x4, 3)
    x6 = astuple(x5, x5)
    x7 = crop(I, ORIGIN, x6)
    x8 = partition(x7)
    x9 = matcher(color, 0)
    x10 = compose(flip, x9)
    x11 = extract(x8, x10)
    x12 = initset(x2)
    x13 = palette(x3)
    x14 = palette(x11)
    x15 = difference(x13, x14)
    x16 = difference(x15, x12)
    x17 = first(x16)
    x18 = interval(0, 3, 1)
    x19 = product(x18, x18)
    x20 = totuple(x19)
    x21 = apply(first, x20)
    x22 = apply(last, x20)
    x23 = lbind(multiply, x5)
    x24 = apply(x23, x21)
    x25 = apply(x23, x22)
    x26 = papply(add, x24, x21)
    x27 = papply(add, x25, x22)
    x28 = papply(astuple, x26, x27)
    x29 = lbind(shift, x11)
    x30 = mapply(x29, x28)
    O = underfill(I, x17, x30)
    return O


def solve_28e73c20(I):
    x1 = width(I)
    x2 = astuple(1, 2)
    x3 = astuple(2, 2)
    x4 = astuple(2, 1)
    x5 = astuple(3, 1)
    x6 = canvas(3, UNITY)
    x7 = upscale(x6, 4)
    x8 = initset(DOWN)
    x9 = insert(UNITY, x8)
    x10 = insert(x2, x9)
    x11 = insert(x3, x10)
    x12 = fill(x7, 0, x11)
    x13 = vupscale(x6, 5)
    x14 = hupscale(x13, 3)
    x15 = insert(x4, x9)
    x16 = insert(x5, x15)
    x17 = fill(x14, 0, x16)
    x18 = even(x1)
    x19 = branch(x18, x12, x17)
    x20 = canvas(0, UNITY)
    x21 = lbind(hupscale, x20)
    x22 = chain(x21, decrement, height)
    x23 = rbind(hconcat, x6)
    x24 = compose(x23, x22)
    x25 = lbind(hupscale, x6)
    x26 = compose(x25, height)
    x27 = fork(vconcat, x24, rot90)
    x28 = fork(vconcat, x26, x27)
    x29 = subtract(x1, 4)
    x30 = power(x28, x29)
    O = x30(x19)
    return O


def solve_4c5c2cf0(I):
    x1 = objects(I, T, T, T)
    x2 = objects(I, F, T, T)
    x3 = first(x2)
    x4 = rbind(subgrid, I)
    x5 = fork(equality, identity, rot90)
    x6 = compose(x5, x4)
    x7 = extract(x1, x6)
    x8 = center(x7)
    x9 = subgrid(x3, I)
    x10 = hmirror(x9)
    x11 = objects(x10, F, T, T)
    x12 = first(x11)
    x13 = objects(x10, T, T, T)
    x14 = rbind(subgrid, x10)
    x15 = compose(x5, x14)
    x16 = extract(x13, x15)
    x17 = center(x16)
    x18 = subtract(x8, x17)
    x19 = shift(x12, x18)
    x20 = paint(I, x19)
    x21 = objects(x20, F, T, T)
    x22 = first(x21)
    x23 = subgrid(x22, x20)
    x24 = vmirror(x23)
    x25 = objects(x24, F, T, T)
    x26 = first(x25)
    x27 = objects(x24, T, T, T)
    x28 = color(x7)
    x29 = matcher(color, x28)
    x30 = extract(x27, x29)
    x31 = center(x30)
    x32 = subtract(x8, x31)
    x33 = shift(x26, x32)
    O = paint(x20, x33)
    return O


def solve_508bd3b6(I):
    x1 = width(I)
    x2 = objects(I, T, T, T)
    x3 = argmin(x2, size)
    x4 = argmax(x2, size)
    x5 = ulcorner(x3)
    x6 = urcorner(x3)
    x7 = index(I, x5)
    x8 = equality(x7, 8)
    x9 = branch(x8, x5, x6)
    x10 = branch(x8, UNITY, DOWN_LEFT)
    x11 = multiply(x10, x1)
    x12 = double(x11)
    x13 = add(x9, x12)
    x14 = subtract(x9, x12)
    x15 = connect(x13, x14)
    x16 = fill(I, 3, x15)
    x17 = paint(x16, x4)
    x18 = objects(x17, T, F, T)
    x19 = rbind(adjacent, x4)
    x20 = extract(x18, x19)
    x21 = first(x20)
    x22 = last(x21)
    x23 = flip(x8)
    x24 = branch(x23, UNITY, DOWN_LEFT)
    x25 = multiply(x24, x1)
    x26 = double(x25)
    x27 = add(x22, x26)
    x28 = subtract(x22, x26)
    x29 = connect(x27, x28)
    x30 = fill(x17, 3, x29)
    x31 = paint(x30, x3)
    O = paint(x31, x4)
    return O


def solve_6d0160f0(I):
    x1 = ofcolor(I, 4)
    x2 = first(x1)
    x3 = first(x2)
    x4 = last(x2)
    x5 = greater(x3, 3)
    x6 = greater(x3, 7)
    x7 = greater(x4, 3)
    x8 = greater(x4, 7)
    x9 = branch(x5, 4, 0)
    x10 = branch(x6, 8, x9)
    x11 = branch(x7, 4, 0)
    x12 = branch(x8, 8, x11)
    x13 = astuple(x10, x12)
    x14 = initset(0)
    x15 = insert(4, x14)
    x16 = insert(8, x15)
    x17 = product(x16, x16)
    x18 = crop(I, ORIGIN, G3x3)
    x19 = asindices(x18)
    x20 = recolor(0, x19)
    x21 = lbind(shift, x20)
    x22 = mapply(x21, x17)
    x23 = paint(I, x22)
    x24 = crop(I, x13, G3x3)
    x25 = replace(x24, 5, 0)
    x26 = ofcolor(x25, 4)
    x27 = first(x26)
    x28 = asindices(x25)
    x29 = toobject(x28, x25)
    x30 = multiply(x27, 4)
    x31 = shift(x29, x30)
    O = paint(x23, x31)
    return O


def solve_f8a8fe49(I):
    x1 = objects(I, T, F, T)
    x2 = replace(I, 5, 0)
    x3 = colorfilter(x1, 2)
    x4 = first(x3)
    x5 = portrait(x4)
    x6 = branch(x5, hsplit, vsplit)
    x7 = branch(x5, vmirror, hmirror)
    x8 = ofcolor(I, 2)
    x9 = subgrid(x8, I)
    x10 = trim(x9)
    x11 = x7(x10)
    x12 = x6(x11, 2)
    x13 = compose(normalize, asobject)
    x14 = apply(x13, x12)
    x15 = last(x14)
    x16 = first(x14)
    x17 = ulcorner(x8)
    x18 = increment(x17)
    x19 = shift(x15, x18)
    x20 = shift(x16, x18)
    x21 = branch(x5, width, height)
    x22 = branch(x5, tojvec, toivec)
    x23 = x21(x15)
    x24 = double(x23)
    x25 = compose(x22, increment)
    x26 = x25(x23)
    x27 = invert(x26)
    x28 = x25(x24)
    x29 = shift(x19, x27)
    x30 = shift(x20, x28)
    x31 = paint(x2, x29)
    O = paint(x31, x30)
    return O


def solve_d07ae81c(I):
    x1 = objects(I, T, F, F)
    x2 = sizefilter(x1, 1)
    x3 = apply(color, x2)
    x4 = difference(x1, x2)
    x5 = apply(color, x4)
    x6 = first(x5)
    x7 = last(x5)
    x8 = ofcolor(I, x6)
    x9 = ofcolor(I, x7)
    x10 = rbind(shoot, UNITY)
    x11 = rbind(shoot, NEG_UNITY)
    x12 = rbind(shoot, DOWN_LEFT)
    x13 = rbind(shoot, UP_RIGHT)
    x14 = fork(combine, x10, x11)
    x15 = fork(combine, x12, x13)
    x16 = fork(combine, x14, x15)
    x17 = compose(x16, center)
    x18 = mapply(x17, x2)
    x19 = intersection(x8, x18)
    x20 = intersection(x9, x18)
    x21 = first(x2)
    x22 = color(x21)
    x23 = center(x21)
    x24 = neighbors(x23)
    x25 = toobject(x24, I)
    x26 = mostcolor(x25)
    x27 = other(x3, x22)
    x28 = equality(x26, x6)
    x29 = branch(x28, x22, x27)
    x30 = branch(x28, x27, x22)
    x31 = fill(I, x29, x19)
    O = fill(x31, x30, x20)
    return O


def solve_6a1e5592(I):
    x1 = width(I)
    x2 = objects(I, T, F, T)
    x3 = astuple(5, x1)
    x4 = crop(I, ORIGIN, x3)
    x5 = colorfilter(x2, 5)
    x6 = merge(x5)
    x7 = cover(I, x6)
    x8 = compose(toindices, normalize)
    x9 = apply(x8, x5)
    x10 = asindices(x4)
    x11 = ofcolor(x4, 0)
    x12 = ofcolor(x4, 2)
    x13 = rbind(multiply, 10)
    x14 = rbind(multiply, 5)
    x15 = rbind(intersection, x12)
    x16 = rbind(intersection, x11)
    x17 = rbind(intersection, x10)
    x18 = chain(x13, size, x15)
    x19 = chain(size, x16, delta)
    x20 = compose(x14, uppermost)
    x21 = chain(size, x16, outbox)
    x22 = chain(x13, size, x17)
    x23 = compose(invert, x18)
    x24 = fork(add, x22, x23)
    x25 = fork(subtract, x24, x21)
    x26 = fork(subtract, x25, x20)
    x27 = fork(subtract, x26, x19)
    x28 = rbind(apply, x10)
    x29 = lbind(lbind, shift)
    x30 = rbind(argmax, x27)
    x31 = chain(x30, x28, x29)
    x32 = mapply(x31, x9)
    O = fill(x7, 1, x32)
    return O


def solve_0e206a2e(I):
    x1 = palette(I)
    x2 = objects(I, F, F, T)
    x3 = rbind(greater, 1)
    x4 = compose(x3, numcolors)
    x5 = sfilter(x2, x4)
    x6 = remove(0, x1)
    x7 = lbind(colorcount, I)
    x8 = argmax(x6, x7)
    x9 = remove(x8, x6)
    x10 = rbind(contained, x9)
    x11 = compose(x10, first)
    x12 = rbind(sfilter, x11)
    x13 = lbind(rbind, subtract)
    x14 = lbind(occurrences, I)
    x15 = lbind(lbind, shift)
    x16 = compose(x13, ulcorner)
    x17 = chain(x16, x12, normalize)
    x18 = chain(x14, x12, normalize)
    x19 = fork(apply, x17, x18)
    x20 = compose(x15, normalize)
    x21 = fork(mapply, x20, x19)
    x22 = astuple(cmirror, dmirror)
    x23 = astuple(hmirror, vmirror)
    x24 = combine(x22, x23)
    x25 = product(x24, x24)
    x26 = fork(compose, first, last)
    x27 = apply(x26, x25)
    x28 = totuple(x27)
    x29 = combine(x24, x28)
    x30 = lbind(rapply, x29)
    x31 = mapply(x30, x5)
    x32 = mapply(x21, x31)
    x33 = paint(I, x32)
    x34 = merge(x5)
    O = cover(x33, x34)
    return O


def solve_d22278a0(I):
    x1 = asindices(I)
    x2 = objects(I, T, F, T)
    x3 = fork(multiply, sign, identity)
    x4 = lbind(apply, x3)
    x5 = chain(even, maximum, x4)
    x6 = lbind(sfilter, x1)
    x7 = fork(add, first, last)
    x8 = rbind(remove, x2)
    x9 = compose(center, last)
    x10 = fork(subtract, first, x9)
    x11 = compose(x5, x10)
    x12 = lbind(rbind, equality)
    x13 = lbind(argmin, x2)
    x14 = chain(x7, x4, x10)
    x15 = lbind(lbind, astuple)
    x16 = lbind(rbind, astuple)
    x17 = lbind(compose, x11)
    x18 = lbind(compose, x14)
    x19 = compose(x18, x15)
    x20 = compose(x18, x16)
    x21 = compose(x13, x19)
    x22 = rbind(compose, x21)
    x23 = lbind(lbind, valmin)
    x24 = rbind(compose, x19)
    x25 = chain(x24, x23, x8)
    x26 = lbind(fork, greater)
    x27 = fork(x26, x25, x20)
    x28 = chain(x6, x17, x16)
    x29 = chain(x6, x22, x12)
    x30 = fork(intersection, x28, x29)
    x31 = compose(x6, x27)
    x32 = fork(intersection, x30, x31)
    x33 = fork(recolor, color, x32)
    x34 = mapply(x33, x2)
    O = paint(I, x34)
    return O


def solve_4290ef0e(I):
    x1 = mostcolor(I)
    x2 = fgpartition(I)
    x3 = objects(I, T, F, T)
    x4 = rbind(valmax, width)
    x5 = lbind(colorfilter, x3)
    x6 = compose(x5, color)
    x7 = compose(double, x4)
    x8 = lbind(prapply, manhattan)
    x9 = fork(x8, identity, identity)
    x10 = lbind(remove, 0)
    x11 = compose(x10, x9)
    x12 = rbind(branch, -2)
    x13 = fork(x12, positive, decrement)
    x14 = chain(x13, minimum, x11)
    x15 = fork(add, x14, x7)
    x16 = compose(x15, x6)
    x17 = compose(invert, x16)
    x18 = order(x2, x17)
    x19 = rbind(argmin, centerofmass)
    x20 = compose(initset, vmirror)
    x21 = fork(insert, dmirror, x20)
    x22 = fork(insert, cmirror, x21)
    x23 = fork(insert, hmirror, x22)
    x24 = compose(x19, x23)
    x25 = apply(x24, x18)
    x26 = size(x2)
    x27 = apply(size, x2)
    x28 = contained(1, x27)
    x29 = increment(x26)
    x30 = branch(x28, x26, x29)
    x31 = double(x30)
    x32 = decrement(x31)
    x33 = apply(normalize, x25)
    x34 = interval(0, x30, 1)
    x35 = pair(x34, x34)
    x36 = mpapply(shift, x33, x35)
    x37 = astuple(x32, x32)
    x38 = canvas(x1, x37)
    x39 = paint(x38, x36)
    x40 = rot90(x39)
    x41 = paint(x40, x36)
    x42 = rot90(x41)
    x43 = paint(x42, x36)
    x44 = rot90(x43)
    O = paint(x44, x36)
    return O


def solve_50846271(I):
    x1 = ofcolor(I, 2)
    x2 = prapply(connect, x1, x1)
    x3 = lbind(greater, 6)
    x4 = compose(x3, size)
    x5 = fork(either, vline, hline)
    x6 = fork(both, x4, x5)
    x7 = mfilter(x2, x6)
    x8 = fill(I, 2, x7)
    x9 = objects(x8, T, F, F)
    x10 = colorfilter(x9, 2)
    x11 = valmax(x10, width)
    x12 = halve(x11)
    x13 = toivec(x12)
    x14 = tojvec(x12)
    x15 = rbind(add, G0x2)
    x16 = rbind(add, G2x0)
    x17 = rbind(subtract, G0x2)
    x18 = rbind(subtract, G2x0)
    x19 = rbind(colorcount, 2)
    x20 = rbind(toobject, x8)
    x21 = compose(initset, x15)
    x22 = fork(insert, x16, x21)
    x23 = fork(insert, x17, x22)
    x24 = fork(insert, x18, x23)
    x25 = fork(combine, dneighbors, x24)
    x26 = chain(x19, x20, x25)
    x27 = rbind(argmax, x26)
    x28 = compose(x27, toindices)
    x29 = apply(x28, x10)
    x30 = rbind(add, x13)
    x31 = rbind(subtract, x13)
    x32 = rbind(add, x14)
    x33 = rbind(subtract, x14)
    x34 = fork(connect, x30, x31)
    x35 = fork(connect, x32, x33)
    x36 = fork(combine, x34, x35)
    x37 = mapply(x36, x29)
    x38 = fill(x8, 8, x37)
    O = fill(x38, 2, x1)
    return O


def solve_b527c5c6(I):
    x1 = objects(I, F, F, T)
    x2 = matcher(first, 2)
    x3 = rbind(sfilter, x2)
    x4 = compose(lowermost, x3)
    x5 = compose(rightmost, x3)
    x6 = compose(uppermost, x3)
    x7 = compose(leftmost, x3)
    x8 = fork(equality, x4, lowermost)
    x9 = fork(equality, x5, rightmost)
    x10 = fork(equality, x6, uppermost)
    x11 = fork(equality, x7, leftmost)
    x12 = compose(invert, x10)
    x13 = compose(invert, x11)
    x14 = fork(add, x12, x8)
    x15 = fork(add, x13, x9)
    x16 = fork(astuple, x14, x15)
    x17 = compose(center, x3)
    x18 = fork(shoot, x17, x16)
    x19 = mapply(x18, x1)
    x20 = fill(I, 2, x19)
    x21 = compose(vline, x18)
    x22 = sfilter(x1, x21)
    x23 = difference(x1, x22)
    x24 = chain(decrement, minimum, shape)
    x25 = compose(increment, x24)
    x26 = compose(invert, x24)
    x27 = rbind(interval, 1)
    x28 = fork(x27, x26, x25)
    x29 = lbind(apply, toivec)
    x30 = lbind(apply, tojvec)
    x31 = lbind(lbind, shift)
    x32 = compose(x31, x18)
    x33 = compose(x29, x28)
    x34 = compose(x30, x28)
    x35 = fork(mapply, x32, x33)
    x36 = fork(mapply, x32, x34)
    x37 = mapply(x35, x23)
    x38 = mapply(x36, x22)
    x39 = combine(x37, x38)
    O = underfill(x20, 3, x39)
    return O


def solve_150deff5(I):
    x1 = canvas(5, G2x2)
    x2 = asobject(x1)
    x3 = occurrences(I, x2)
    x4 = lbind(shift, x2)
    x5 = mapply(x4, x3)
    x6 = fill(I, 8, x5)
    x7 = canvas(5, UNITY)
    x8 = astuple(2, 1)
    x9 = canvas(8, x8)
    x10 = vconcat(x9, x7)
    x11 = asobject(x10)
    x12 = occurrences(x6, x11)
    x13 = lbind(shift, x11)
    x14 = mapply(x13, x12)
    x15 = fill(x6, 2, x14)
    x16 = astuple(1, 3)
    x17 = canvas(5, x16)
    x18 = asobject(x17)
    x19 = occurrences(x15, x18)
    x20 = lbind(shift, x18)
    x21 = mapply(x20, x19)
    x22 = fill(x15, 2, x21)
    x23 = hmirror(x10)
    x24 = asobject(x23)
    x25 = occurrences(x22, x24)
    x26 = lbind(shift, x24)
    x27 = mapply(x26, x25)
    x28 = fill(x22, 2, x27)
    x29 = dmirror(x10)
    x30 = asobject(x29)
    x31 = occurrences(x28, x30)
    x32 = lbind(shift, x30)
    x33 = mapply(x32, x31)
    x34 = fill(x28, 2, x33)
    x35 = vmirror(x29)
    x36 = asobject(x35)
    x37 = occurrences(x34, x36)
    x38 = lbind(shift, x36)
    x39 = mapply(x38, x37)
    O = fill(x34, 2, x39)
    return O


def solve_b7249182(I):
    x1 = objects(I, T, F, T)
    x2 = merge(x1)
    x3 = portrait(x2)
    x4 = branch(x3, identity, dmirror)
    x5 = x4(I)
    x6 = objects(x5, T, F, T)
    x7 = order(x6, uppermost)
    x8 = first(x7)
    x9 = last(x7)
    x10 = color(x8)
    x11 = color(x9)
    x12 = compose(first, toindices)
    x13 = x12(x8)
    x14 = x12(x9)
    x15 = connect(x13, x14)
    x16 = centerofmass(x15)
    x17 = connect(x13, x16)
    x18 = fill(x5, x11, x15)
    x19 = fill(x18, x10, x17)
    x20 = add(x16, DOWN)
    x21 = initset(x16)
    x22 = insert(x20, x21)
    x23 = toobject(x22, x19)
    x24 = astuple(0, -2)
    x25 = shift(x23, G0x2)
    x26 = shift(x23, x24)
    x27 = combine(x25, x26)
    x28 = ulcorner(x27)
    x29 = urcorner(x27)
    x30 = connect(x28, x29)
    x31 = shift(x30, UP)
    x32 = llcorner(x27)
    x33 = lrcorner(x27)
    x34 = connect(x32, x33)
    x35 = shift(x34, DOWN)
    x36 = paint(x19, x27)
    x37 = fill(x36, x10, x31)
    x38 = fill(x37, x11, x35)
    x39 = cover(x38, x22)
    O = x4(x39)
    return O


def solve_9d9215db(I):
    x1 = rot90(I)
    x2 = rot180(I)
    x3 = rot270(I)
    x4 = initset(I)
    x5 = chain(numcolors, lefthalf, tophalf)
    x6 = insert(x1, x4)
    x7 = insert(x2, x6)
    x8 = insert(x3, x7)
    x9 = argmax(x8, x5)
    x10 = vmirror(x9)
    x11 = papply(pair, x9, x10)
    x12 = lbind(apply, maximum)
    x13 = apply(x12, x11)
    x14 = partition(x13)
    x15 = sizefilter(x14, 4)
    x16 = apply(llcorner, x15)
    x17 = apply(lrcorner, x15)
    x18 = combine(x16, x17)
    x19 = cover(x13, x18)
    x20 = tojvec(-2)
    x21 = rbind(add, G0x2)
    x22 = rbind(add, x20)
    x23 = compose(x21, ulcorner)
    x24 = compose(x22, urcorner)
    x25 = fork(connect, x23, x24)
    x26 = compose(even, last)
    x27 = rbind(sfilter, x26)
    x28 = chain(normalize, x27, x25)
    x29 = fork(shift, x28, x23)
    x30 = fork(recolor, color, x29)
    x31 = mapply(x30, x15)
    x32 = paint(x19, x31)
    x33 = rot90(x32)
    x34 = rot180(x32)
    x35 = rot270(x32)
    x36 = papply(pair, x32, x33)
    x37 = apply(x12, x36)
    x38 = papply(pair, x37, x34)
    x39 = apply(x12, x38)
    x40 = papply(pair, x39, x35)
    O = apply(x12, x40)
    return O


def solve_6855a6e4(I):
    x1 = fgpartition(I)
    x2 = rot90(I)
    x3 = colorfilter(x1, 2)
    x4 = first(x3)
    x5 = portrait(x4)
    x6 = branch(x5, I, x2)
    x7 = objects(x6, T, F, T)
    x8 = colorfilter(x7, 5)
    x9 = apply(center, x8)
    x10 = valmin(x9, first)
    x11 = compose(first, center)
    x12 = matcher(x11, x10)
    x13 = compose(flip, x12)
    x14 = extract(x8, x12)
    x15 = extract(x8, x13)
    x16 = ulcorner(x14)
    x17 = ulcorner(x15)
    x18 = subgrid(x14, x6)
    x19 = subgrid(x15, x6)
    x20 = hmirror(x18)
    x21 = hmirror(x19)
    x22 = ofcolor(x20, 5)
    x23 = recolor(5, x22)
    x24 = ofcolor(x21, 5)
    x25 = recolor(5, x24)
    x26 = height(x23)
    x27 = height(x25)
    x28 = add(3, x26)
    x29 = add(3, x27)
    x30 = toivec(x28)
    x31 = toivec(x29)
    x32 = add(x16, x30)
    x33 = subtract(x17, x31)
    x34 = shift(x23, x32)
    x35 = shift(x25, x33)
    x36 = merge(x8)
    x37 = cover(x6, x36)
    x38 = paint(x37, x34)
    x39 = paint(x38, x35)
    x40 = rot270(x39)
    O = branch(x5, x39, x40)
    return O


def solve_264363fd(I):
    x1 = objects(I, F, F, T)
    x2 = argmin(x1, size)
    x3 = normalize(x2)
    x4 = height(x2)
    x5 = width(x2)
    x6 = equality(x4, 5)
    x7 = equality(x5, 5)
    x8 = astuple(x6, x7)
    x9 = add(UNITY, x8)
    x10 = invert(x9)
    x11 = center(x2)
    x12 = index(I, x11)
    x13 = branch(x6, UP, RIGHT)
    x14 = add(x13, x11)
    x15 = index(I, x14)
    x16 = astuple(x12, ORIGIN)
    x17 = initset(x16)
    x18 = cover(I, x2)
    x19 = mostcolor(x18)
    x20 = ofcolor(x18, x19)
    x21 = occurrences(x18, x17)
    x22 = objects(x18, F, F, T)
    x23 = rbind(occurrences, x17)
    x24 = rbind(subgrid, x18)
    x25 = compose(x23, x24)
    x26 = lbind(mapply, vfrontier)
    x27 = lbind(mapply, hfrontier)
    x28 = compose(x26, x25)
    x29 = compose(x27, x25)
    x30 = branch(x6, x28, x29)
    x31 = branch(x7, x29, x28)
    x32 = fork(combine, x30, x31)
    x33 = lbind(recolor, x15)
    x34 = compose(x33, x32)
    x35 = fork(paint, x24, x34)
    x36 = compose(asobject, x35)
    x37 = fork(shift, x36, ulcorner)
    x38 = mapply(x37, x22)
    x39 = paint(x18, x38)
    x40 = shift(x3, x10)
    x41 = lbind(shift, x40)
    x42 = mapply(x41, x21)
    x43 = paint(x39, x42)
    O = fill(x43, x19, x20)
    return O


def solve_7df24a62(I):
    x1 = height(I)
    x2 = width(I)
    x3 = ofcolor(I, 1)
    x4 = ofcolor(I, 4)
    x5 = ulcorner(x3)
    x6 = subgrid(x3, I)
    x7 = rot90(x6)
    x8 = rot180(x6)
    x9 = rot270(x6)
    x10 = matcher(size, 0)
    x11 = rbind(ofcolor, 1)
    x12 = compose(normalize, x11)
    x13 = rbind(ofcolor, 4)
    x14 = rbind(shift, x5)
    x15 = compose(x14, x13)
    x16 = lbind(subtract, x1)
    x17 = chain(increment, x16, height)
    x18 = lbind(subtract, x2)
    x19 = chain(increment, x18, width)
    x20 = rbind(interval, 1)
    x21 = lbind(x20, 0)
    x22 = compose(x21, x17)
    x23 = compose(x21, x19)
    x24 = fork(product, x22, x23)
    x25 = rbind(shift, NEG_UNITY)
    x26 = lbind(lbind, shift)
    x27 = chain(x26, x25, x12)
    x28 = astuple(x6, x7)
    x29 = astuple(x8, x9)
    x30 = combine(x28, x29)
    x31 = apply(x15, x30)
    x32 = lbind(difference, x4)
    x33 = apply(x32, x31)
    x34 = apply(normalize, x31)
    x35 = apply(x24, x34)
    x36 = lbind(rbind, difference)
    x37 = apply(x26, x34)
    x38 = apply(x36, x33)
    x39 = papply(compose, x38, x37)
    x40 = lbind(compose, x10)
    x41 = apply(x40, x39)
    x42 = papply(sfilter, x35, x41)
    x43 = apply(x27, x30)
    x44 = mpapply(mapply, x43, x42)
    O = fill(I, 1, x44)
    return O


def solve_f15e1fac(I):
    x1 = ofcolor(I, 2)
    x2 = portrait(x1)
    x3 = branch(x2, identity, dmirror)
    x4 = x3(I)
    x5 = leftmost(x1)
    x6 = equality(x5, 0)
    x7 = branch(x6, identity, vmirror)
    x8 = x7(x4)
    x9 = ofcolor(x8, 8)
    x10 = uppermost(x9)
    x11 = equality(x10, 0)
    x12 = branch(x11, identity, hmirror)
    x13 = x12(x8)
    x14 = ofcolor(x13, 8)
    x15 = ofcolor(x13, 2)
    x16 = rbind(shoot, DOWN)
    x17 = mapply(x16, x14)
    x18 = height(x13)
    x19 = apply(first, x15)
    x20 = insert(0, x19)
    x21 = insert(x18, x19)
    x22 = apply(decrement, x21)
    x23 = order(x20, identity)
    x24 = order(x22, identity)
    x25 = size(x15)
    x26 = increment(x25)
    x27 = interval(0, x26, 1)
    x28 = apply(tojvec, x27)
    x29 = pair(x23, x24)
    x30 = lbind(sfilter, x17)
    x31 = compose(first, last)
    x32 = chain(decrement, first, first)
    x33 = fork(greater, x31, x32)
    x34 = chain(increment, last, first)
    x35 = fork(greater, x34, x31)
    x36 = fork(both, x33, x35)
    x37 = lbind(lbind, astuple)
    x38 = lbind(compose, x36)
    x39 = chain(x30, x38, x37)
    x40 = apply(x39, x29)
    x41 = papply(shift, x40, x28)
    x42 = merge(x41)
    x43 = fill(x13, 8, x42)
    x44 = chain(x3, x7, x12)
    O = x44(x43)
    return O


def solve_234bbc79(I):
    x1 = objects(I, F, F, T)
    x2 = rbind(other, 5)
    x3 = compose(x2, palette)
    x4 = fork(recolor, x3, identity)
    x5 = apply(x4, x1)
    x6 = order(x5, leftmost)
    x7 = compose(last, last)
    x8 = lbind(matcher, x7)
    x9 = compose(x8, leftmost)
    x10 = compose(x8, rightmost)
    x11 = fork(sfilter, identity, x9)
    x12 = fork(sfilter, identity, x10)
    x13 = compose(dneighbors, last)
    x14 = rbind(chain, x13)
    x15 = lbind(x14, size)
    x16 = lbind(rbind, intersection)
    x17 = chain(x15, x16, toindices)
    x18 = fork(argmin, x11, x17)
    x19 = fork(argmin, x12, x17)
    x20 = compose(last, x18)
    x21 = compose(last, x19)
    x22 = astuple(0, DOWN_LEFT)
    x23 = initset(x22)
    x24 = lbind(add, RIGHT)
    x25 = chain(x20, first, last)
    x26 = compose(x21, first)
    x27 = fork(subtract, x26, x25)
    x28 = compose(first, last)
    x29 = compose(x24, x27)
    x30 = fork(shift, x28, x29)
    x31 = fork(combine, first, x30)
    x32 = fork(remove, x28, last)
    x33 = fork(astuple, x31, x32)
    x34 = size(x1)
    x35 = power(x33, x34)
    x36 = astuple(x23, x6)
    x37 = x35(x36)
    x38 = first(x37)
    x39 = width(x38)
    x40 = decrement(x39)
    x41 = astuple(3, x40)
    x42 = canvas(0, x41)
    O = paint(x42, x38)
    return O


def solve_22233c11(I):
    x1 = objects(I, T, T, T)
    x2 = rbind(upscale, 2)
    x3 = chain(invert, halve, shape)
    x4 = fork(combine, hfrontier, vfrontier)
    x5 = compose(x2, vmirror)
    x6 = fork(shift, x5, x3)
    x7 = compose(toindices, x6)
    x8 = lbind(mapply, x4)
    x9 = compose(x8, toindices)
    x10 = fork(difference, x7, x9)
    x11 = mapply(x10, x1)
    O = fill(I, 8, x11)
    return O


def solve_2dd70a9a(I):
    x1 = ofcolor(I, 2)
    x2 = ofcolor(I, 3)
    x3 = vline(x1)
    x4 = vline(x2)
    x5 = center(x1)
    x6 = branch(x4, uppermost, rightmost)
    x7 = x6(x1)
    x8 = x6(x2)
    x9 = greater(x7, x8)
    x10 = both(x4, x9)
    x11 = branch(x10, lowermost, uppermost)
    x12 = x11(x2)
    x13 = branch(x4, leftmost, rightmost)
    x14 = x13(x2)
    x15 = astuple(x12, x14)
    x16 = other(x2, x15)
    x17 = subtract(x15, x16)
    x18 = shoot(x15, x17)
    x19 = underfill(I, 1, x18)
    x20 = objects(x19, T, F, F)
    x21 = colorfilter(x20, 1)
    x22 = rbind(adjacent, x2)
    x23 = sfilter(x21, x22)
    x24 = difference(x21, x23)
    x25 = merge(x24)
    x26 = cover(x19, x25)
    x27 = shoot(x5, DOWN)
    x28 = shoot(x5, UP)
    x29 = shoot(x5, LEFT)
    x30 = shoot(x5, RIGHT)
    x31 = combine(x27, x28)
    x32 = combine(x29, x30)
    x33 = branch(x3, x31, x32)
    x34 = ofcolor(x26, 1)
    x35 = initset(x15)
    x36 = rbind(manhattan, x35)
    x37 = compose(x36, initset)
    x38 = argmax(x34, x37)
    x39 = initset(x38)
    x40 = gravitate(x39, x33)
    x41 = crement(x40)
    x42 = add(x38, x41)
    x43 = connect(x38, x42)
    x44 = fill(x26, 1, x43)
    x45 = connect(x42, x5)
    x46 = underfill(x44, 1, x45)
    O = replace(x46, 1, 3)
    return O


def solve_a64e4611(I):
    x1 = asindices(I)
    x2 = fork(product, identity, identity)
    x3 = lbind(canvas, 0)
    x4 = compose(asobject, x3)
    x5 = fork(multiply, first, last)
    x6 = compose(positive, size)
    x7 = lbind(lbind, shift)
    x8 = rbind(fork, x5)
    x9 = lbind(x8, multiply)
    x10 = lbind(chain, x6)
    x11 = rbind(x10, x4)
    x12 = lbind(lbind, occurrences)
    x13 = chain(x9, x11, x12)
    x14 = compose(x2, first)
    x15 = compose(x13, last)
    x16 = fork(argmax, x14, x15)
    x17 = chain(x7, x4, x16)
    x18 = compose(x4, x16)
    x19 = fork(occurrences, last, x18)
    x20 = fork(mapply, x17, x19)
    x21 = multiply(2, 6)
    x22 = interval(3, x21, 1)
    x23 = astuple(x22, I)
    x24 = x20(x23)
    x25 = fill(I, 3, x24)
    x26 = interval(3, 10, 1)
    x27 = astuple(x26, x25)
    x28 = x20(x27)
    x29 = fill(x25, 3, x28)
    x30 = astuple(x26, x29)
    x31 = x20(x30)
    x32 = fill(x29, 3, x31)
    x33 = rbind(toobject, x32)
    x34 = rbind(colorcount, 3)
    x35 = chain(x34, x33, neighbors)
    x36 = matcher(x35, 8)
    x37 = sfilter(x1, x36)
    x38 = fill(I, 3, x37)
    x39 = ofcolor(x38, 0)
    x40 = rbind(bordering, x38)
    x41 = compose(x40, initset)
    x42 = lbind(contained, 3)
    x43 = rbind(toobject, x38)
    x44 = chain(x42, palette, x43)
    x45 = compose(x44, dneighbors)
    x46 = fork(both, x45, x41)
    x47 = sfilter(x39, x46)
    O = fill(x38, 3, x47)
    return O


def solve_7837ac64(I):
    x1 = fgpartition(I)
    x2 = argmax(x1, size)
    x3 = remove(x2, x1)
    x4 = merge(x3)
    x5 = subgrid(x4, I)
    x6 = chain(color, merge, frontiers)
    x7 = x6(I)
    x8 = objects(x5, T, F, F)
    x9 = colorfilter(x8, 0)
    x10 = rbind(toobject, x5)
    x11 = chain(x10, corners, outbox)
    x12 = lbind(contained, x7)
    x13 = chain(x12, palette, x11)
    x14 = compose(numcolors, x11)
    x15 = compose(flip, x13)
    x16 = matcher(x14, 1)
    x17 = fork(both, x15, x16)
    x18 = sfilter(x9, x17)
    x19 = compose(color, x11)
    x20 = fork(recolor, x19, identity)
    x21 = mapply(x20, x18)
    x22 = paint(x5, x21)
    x23 = first(x9)
    x24 = height(x23)
    x25 = height(x5)
    x26 = increment(x24)
    x27 = interval(0, x25, x26)
    x28 = interval(0, x25, 1)
    x29 = rbind(contained, x27)
    x30 = chain(flip, x29, last)
    x31 = lbind(apply, first)
    x32 = rbind(sfilter, x30)
    x33 = rbind(pair, x28)
    x34 = chain(x31, x32, x33)
    x35 = compose(dmirror, x34)
    x36 = power(x35, 2)
    x37 = x36(x22)
    O = downscale(x37, x24)
    return O


def solve_a8c38be5(I):
    x1 = replace(I, 5, 0)
    x2 = objects(x1, T, F, T)
    x3 = apply(normalize, x2)
    x4 = astuple(9, 9)
    x5 = canvas(5, x4)
    x6 = asindices(x5)
    x7 = box(x6)
    x8 = center(x6)
    x9 = lbind(contained, 0)
    x10 = rbind(subtract, x8)
    x11 = compose(x9, x10)
    x12 = chain(outbox, outbox, initset)
    x13 = corners(x6)
    x14 = mapply(x12, x13)
    x15 = difference(x7, x14)
    x16 = inbox(x7)
    x17 = sfilter(x16, x11)
    x18 = combine(x15, x17)
    x19 = fill(x5, 1, x18)
    x20 = objects(x19, T, F, T)
    x21 = apply(toindices, x20)
    x22 = lbind(matcher, normalize)
    x23 = lbind(extract, x21)
    x24 = chain(ulcorner, x23, x22)
    x25 = compose(x24, toindices)
    x26 = fork(shift, identity, x25)
    x27 = mapply(x26, x3)
    O = paint(x5, x27)
    return O


def solve_b775ac94(I):
    x1 = objects(I, F, T, T)
    x2 = lbind(rbind, equality)
    x3 = rbind(compose, first)
    x4 = chain(x3, x2, mostcolor)
    x5 = fork(sfilter, identity, x4)
    x6 = fork(difference, identity, x5)
    x7 = lbind(rbind, adjacent)
    x8 = rbind(compose, initset)
    x9 = chain(x8, x7, x6)
    x10 = fork(extract, x5, x9)
    x11 = fork(insert, x10, x6)
    x12 = lbind(recolor, 0)
    x13 = chain(x12, delta, x11)
    x14 = fork(combine, x11, x13)
    x15 = fork(position, x5, x6)
    x16 = chain(toivec, first, x15)
    x17 = chain(tojvec, last, x15)
    x18 = fork(multiply, shape, x16)
    x19 = fork(multiply, shape, x17)
    x20 = fork(multiply, shape, x15)
    x21 = fork(shift, hmirror, x18)
    x22 = fork(shift, vmirror, x19)
    x23 = compose(hmirror, vmirror)
    x24 = fork(shift, x23, x20)
    x25 = lbind(compose, x5)
    x26 = x25(x21)
    x27 = x25(x22)
    x28 = x25(x24)
    x29 = compose(crement, invert)
    x30 = lbind(compose, x29)
    x31 = x30(x16)
    x32 = x30(x17)
    x33 = x30(x15)
    x34 = fork(shift, x26, x31)
    x35 = fork(shift, x27, x32)
    x36 = fork(shift, x28, x33)
    x37 = lbind(index, I)
    x38 = lbind(compose, toindices)
    x39 = x38(x14)
    x40 = x38(x34)
    x41 = x38(x35)
    x42 = x38(x36)
    x43 = fork(intersection, x39, x40)
    x44 = fork(intersection, x39, x41)
    x45 = fork(intersection, x39, x42)
    x46 = chain(x37, first, x43)
    x47 = chain(x37, first, x44)
    x48 = chain(x37, first, x45)
    x49 = fork(recolor, x46, x34)
    x50 = fork(recolor, x47, x35)
    x51 = fork(recolor, x48, x36)
    x52 = mapply(x49, x1)
    x53 = mapply(x50, x1)
    x54 = mapply(x51, x1)
    x55 = paint(I, x52)
    x56 = paint(x55, x53)
    O = paint(x56, x54)
    return O


def solve_97a05b5b(I):
    x1 = objects(I, F, T, T)
    x2 = argmax(x1, size)
    x3 = subgrid(x2, I)
    x4 = rbind(greater, 1)
    x5 = compose(x4, numcolors)
    x6 = sfilter(x1, x5)
    x7 = lbind(rbind, subtract)
    x8 = switch(x3, 2, 0)
    x9 = lbind(occurrences, x8)
    x10 = lbind(lbind, shift)
    x11 = compose(x7, ulcorner)
    x12 = matcher(first, 2)
    x13 = compose(flip, x12)
    x14 = rbind(sfilter, x12)
    x15 = rbind(sfilter, x13)
    x16 = lbind(recolor, 0)
    x17 = compose(x16, x15)
    x18 = fork(combine, x17, x14)
    x19 = chain(x11, x18, normalize)
    x20 = objects(x8, T, T, T)
    x21 = apply(toindices, x20)
    x22 = chain(x9, x18, normalize)
    x23 = rbind(colorcount, 2)
    x24 = lbind(sfilter, x21)
    x25 = chain(size, first, x24)
    x26 = compose(positive, size)
    x27 = lbind(lbind, contained)
    x28 = chain(x26, x24, x27)
    x29 = compose(x25, x27)
    x30 = rbind(sfilter, x28)
    x31 = compose(x30, x22)
    x32 = lbind(rbind, equality)
    x33 = rbind(compose, x29)
    x34 = chain(x33, x32, x23)
    x35 = fork(sfilter, x31, x34)
    x36 = fork(apply, x19, x35)
    x37 = compose(x10, normalize)
    x38 = fork(mapply, x37, x36)
    x39 = astuple(cmirror, dmirror)
    x40 = astuple(hmirror, vmirror)
    x41 = combine(x39, x40)
    x42 = product(x41, x41)
    x43 = fork(compose, first, last)
    x44 = apply(x43, x42)
    x45 = lbind(rapply, x44)
    x46 = mapply(x45, x6)
    x47 = mapply(x38, x46)
    x48 = paint(x3, x47)
    x49 = palette(x47)
    x50 = lbind(remove, 2)
    x51 = x50(x49)
    x52 = chain(first, x50, palette)
    x53 = rbind(contained, x51)
    x54 = chain(flip, x53, x52)
    x55 = sfilter(x6, x54)
    x56 = fork(apply, x19, x22)
    x57 = fork(mapply, x37, x56)
    x58 = mapply(x45, x55)
    x59 = mapply(x57, x58)
    O = paint(x48, x59)
    return O


def solve_3e980e27(I):
    x1 = objects(I, F, T, T)
    x2 = astuple(10, 10)
    x3 = invert(x2)
    x4 = astuple(2, x3)
    x5 = astuple(3, x3)
    x6 = initset(x4)
    x7 = insert(x5, x6)
    x8 = insert(x7, x1)
    x9 = lbind(contained, 2)
    x10 = lbind(contained, 3)
    x11 = compose(invert, ulcorner)
    x12 = lbind(compose, x11)
    x13 = lbind(rbind, sfilter)
    x14 = compose(x12, x13)
    x15 = rbind(compose, center)
    x16 = lbind(lbind, shift)
    x17 = x14(x9)
    x18 = x14(x10)
    x19 = fork(shift, identity, x17)
    x20 = fork(shift, identity, x18)
    x21 = compose(x9, palette)
    x22 = compose(x10, palette)
    x23 = sfilter(x8, x21)
    x24 = argmax(x23, size)
    x25 = remove(x24, x23)
    x26 = vmirror(x24)
    x27 = chain(x15, x16, x19)
    x28 = x27(x26)
    x29 = mapply(x28, x25)
    x30 = sfilter(x8, x22)
    x31 = argmax(x30, size)
    x32 = remove(x31, x30)
    x33 = chain(x15, x16, x20)
    x34 = x33(x31)
    x35 = mapply(x34, x32)
    x36 = combine(x29, x35)
    O = paint(I, x36)
    return O