Ф¬
з Ї 
:
Add
x"T
y"T
z"T"
Ttype:
2	
Ы
ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
E
AssignAddVariableOp
resource
value"dtype"
dtypetypeИ
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
8
DivNoNan
x"T
y"T
z"T"
Ttype:	
2
h
Equal
x"T
y"T
z
"
Ttype:
2	
"$
incompatible_shape_errorbool(Р
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
L
PreventGradient

input"T
output"T"	
Ttype"
messagestring 
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
а
ResourceApplyAdam
var
m
v
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( И
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
У
#SparseSoftmaxCrossEntropyWithLogits
features"T
labels"Tlabels	
loss"T
backprop"T"
Ttype:
2"
Tlabelstype0	:
2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
ц
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
М
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ
9
VarIsInitializedOp
resource
is_initialized
И
&
	ZerosLike
x"T
y"T"	
Ttype"train*1.15.02v1.15.0-rc3-22-g590d6ee8ЈД
p
dense_inputPlaceholder*
dtype0*(
_output_shapes
:€€€€€€€€€А*
shape:€€€€€€€€€А
Я
-dense/kernel/Initializer/random_uniform/shapeConst*
valueB"   @   *
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:
С
+dense/kernel/Initializer/random_uniform/minConst*
valueB
 *тк≠љ*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
С
+dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *тк≠=*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
Ќ
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	А@*
T0*
_class
loc:@dense/kernel
ќ
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
: 
б
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
_output_shapes
:	А@*
T0*
_class
loc:@dense/kernel
”
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	А@
Ц
dense/kernelVarHandleOp*
shape:	А@*
shared_namedense/kernel*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
i
-dense/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense/kernel*
_output_shapes
: 
k
dense/kernel/AssignAssignVariableOpdense/kernel'dense/kernel/Initializer/random_uniform*
dtype0
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes
:	А@
И
dense/bias/Initializer/zerosConst*
valueB@*    *
_class
loc:@dense/bias*
dtype0*
_output_shapes
:@
Л

dense/biasVarHandleOp*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: *
shape:@*
shared_name
dense/bias
e
+dense/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp
dense/bias*
_output_shapes
: 
\
dense/bias/AssignAssignVariableOp
dense/biasdense/bias/Initializer/zeros*
dtype0
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:@
i
dense/MatMul/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes
:	А@
r
dense/MatMulMatMuldense_inputdense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€@
c
dense/BiasAdd/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:@
v
dense/BiasAddBiasAdddense/MatMuldense/BiasAdd/ReadVariableOp*'
_output_shapes
:€€€€€€€€€@*
T0
S

dense/ReluReludense/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€@
£
/dense_1/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"@      *!
_class
loc:@dense_1/kernel
Х
-dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *2uЖЊ*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
Х
-dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *2uЖ>*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
“
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
T0*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes

:@
÷
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
: 
и
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:@
Џ
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*
_output_shapes

:@*
T0*!
_class
loc:@dense_1/kernel
Ы
dense_1/kernelVarHandleOp*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: *
shape
:@*
shared_namedense_1/kernel
m
/dense_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/kernel*
_output_shapes
: 
q
dense_1/kernel/AssignAssignVariableOpdense_1/kernel)dense_1/kernel/Initializer/random_uniform*
dtype0
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes

:@
М
dense_1/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *
_class
loc:@dense_1/bias
С
dense_1/biasVarHandleOp*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: *
shape:*
shared_namedense_1/bias
i
-dense_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/bias*
_output_shapes
: 
b
dense_1/bias/AssignAssignVariableOpdense_1/biasdense_1/bias/Initializer/zeros*
dtype0
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes
:
l
dense_1/MatMul/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes

:@
u
dense_1/MatMulMatMul
dense/Reludense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€
g
dense_1/BiasAdd/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes
:
|
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€
]
dense_1/SoftmaxSoftmaxdense_1/BiasAdd*'
_output_shapes
:€€€€€€€€€*
T0
Э
0TFOptimizer/iterations/Initializer/initial_valueConst*
value	B	 R *)
_class
loc:@TFOptimizer/iterations*
dtype0	*
_output_shapes
: 
Ђ
TFOptimizer/iterationsVarHandleOp*
dtype0	*
_output_shapes
: *
shape: *'
shared_nameTFOptimizer/iterations*)
_class
loc:@TFOptimizer/iterations
}
7TFOptimizer/iterations/IsInitialized/VarIsInitializedOpVarIsInitializedOpTFOptimizer/iterations*
_output_shapes
: 
И
TFOptimizer/iterations/AssignAssignVariableOpTFOptimizer/iterations0TFOptimizer/iterations/Initializer/initial_value*
dtype0	
y
*TFOptimizer/iterations/Read/ReadVariableOpReadVariableOpTFOptimizer/iterations*
dtype0	*
_output_shapes
: 
Г
dense_1_targetPlaceholder*
dtype0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*%
shape:€€€€€€€€€€€€€€€€€€
v
total/Initializer/zerosConst*
valueB
 *    *
_class

loc:@total*
dtype0*
_output_shapes
: 
x
totalVarHandleOp*
shape: *
shared_nametotal*
_class

loc:@total*
dtype0*
_output_shapes
: 
[
&total/IsInitialized/VarIsInitializedOpVarIsInitializedOptotal*
_output_shapes
: 
M
total/AssignAssignVariableOptotaltotal/Initializer/zeros*
dtype0
W
total/Read/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
v
count/Initializer/zerosConst*
valueB
 *    *
_class

loc:@count*
dtype0*
_output_shapes
: 
x
countVarHandleOp*
_class

loc:@count*
dtype0*
_output_shapes
: *
shape: *
shared_namecount
[
&count/IsInitialized/VarIsInitializedOpVarIsInitializedOpcount*
_output_shapes
: 
M
count/AssignAssignVariableOpcountcount/Initializer/zeros*
dtype0
W
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_output_shapes
: 
|
metrics/acc/SqueezeSqueezedense_1_target*
squeeze_dims

€€€€€€€€€*
T0*#
_output_shapes
:€€€€€€€€€
g
metrics/acc/ArgMax/dimensionConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
y
metrics/acc/ArgMaxArgMaxdense_1/Softmaxmetrics/acc/ArgMax/dimension*
T0*#
_output_shapes
:€€€€€€€€€
i
metrics/acc/CastCastmetrics/acc/ArgMax*

SrcT0	*#
_output_shapes
:€€€€€€€€€*

DstT0
o
metrics/acc/EqualEqualmetrics/acc/Squeezemetrics/acc/Cast*
T0*#
_output_shapes
:€€€€€€€€€
j
metrics/acc/Cast_1Castmetrics/acc/Equal*#
_output_shapes
:€€€€€€€€€*

DstT0*

SrcT0

[
metrics/acc/ConstConst*
valueB: *
dtype0*
_output_shapes
:
^
metrics/acc/SumSummetrics/acc/Cast_1metrics/acc/Const*
T0*
_output_shapes
: 
[
metrics/acc/AssignAddVariableOpAssignAddVariableOptotalmetrics/acc/Sum*
dtype0
М
metrics/acc/ReadVariableOpReadVariableOptotal ^metrics/acc/AssignAddVariableOp^metrics/acc/Sum*
dtype0*
_output_shapes
: 
M
metrics/acc/SizeSizemetrics/acc/Cast_1*
T0*
_output_shapes
: 
\
metrics/acc/Cast_2Castmetrics/acc/Size*

SrcT0*
_output_shapes
: *

DstT0
В
!metrics/acc/AssignAddVariableOp_1AssignAddVariableOpcountmetrics/acc/Cast_2 ^metrics/acc/AssignAddVariableOp*
dtype0
†
metrics/acc/ReadVariableOp_1ReadVariableOpcount ^metrics/acc/AssignAddVariableOp"^metrics/acc/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
З
%metrics/acc/div_no_nan/ReadVariableOpReadVariableOptotal"^metrics/acc/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
Й
'metrics/acc/div_no_nan/ReadVariableOp_1ReadVariableOpcount"^metrics/acc/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
У
metrics/acc/div_no_nanDivNoNan%metrics/acc/div_no_nan/ReadVariableOp'metrics/acc/div_no_nan/ReadVariableOp_1*
_output_shapes
: *
T0
Y
metrics/acc/IdentityIdentitymetrics/acc/div_no_nan*
T0*
_output_shapes
: 
x
loss/dense_1_loss/CastCastdense_1_target*

SrcT0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*

DstT0	
V
loss/dense_1_loss/ShapeShapedense_1/BiasAdd*
T0*
_output_shapes
:
r
loss/dense_1_loss/Reshape/shapeConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
Л
loss/dense_1_loss/ReshapeReshapeloss/dense_1_loss/Castloss/dense_1_loss/Reshape/shape*
T0	*#
_output_shapes
:€€€€€€€€€
x
%loss/dense_1_loss/strided_slice/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
q
'loss/dense_1_loss/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
q
'loss/dense_1_loss/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
З
loss/dense_1_loss/strided_sliceStridedSliceloss/dense_1_loss/Shape%loss/dense_1_loss/strided_slice/stack'loss/dense_1_loss/strided_slice/stack_1'loss/dense_1_loss/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
n
#loss/dense_1_loss/Reshape_1/shape/0Const*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
Э
!loss/dense_1_loss/Reshape_1/shapePack#loss/dense_1_loss/Reshape_1/shape/0loss/dense_1_loss/strided_slice*
N*
_output_shapes
:*
T0
Х
loss/dense_1_loss/Reshape_1Reshapedense_1/BiasAdd!loss/dense_1_loss/Reshape_1/shape*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
T0
Д
;loss/dense_1_loss/SparseSoftmaxCrossEntropyWithLogits/ShapeShapeloss/dense_1_loss/Reshape*
T0	*
_output_shapes
:
В
Yloss/dense_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsloss/dense_1_loss/Reshape_1loss/dense_1_loss/Reshape*
T0*?
_output_shapes-
+:€€€€€€€€€:€€€€€€€€€€€€€€€€€€
k
&loss/dense_1_loss/weighted_loss/Cast/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Ч
Tloss/dense_1_loss/weighted_loss/broadcast_weights/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
Х
Sloss/dense_1_loss/weighted_loss/broadcast_weights/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
№
Sloss/dense_1_loss/weighted_loss/broadcast_weights/assert_broadcastable/values/shapeShapeYloss/dense_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*
_output_shapes
:
Ф
Rloss/dense_1_loss/weighted_loss/broadcast_weights/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
j
bloss/dense_1_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
ѓ
Aloss/dense_1_loss/weighted_loss/broadcast_weights/ones_like/ShapeShapeYloss/dense_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitsc^loss/dense_1_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
_output_shapes
:*
T0
л
Aloss/dense_1_loss/weighted_loss/broadcast_weights/ones_like/ConstConstc^loss/dense_1_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
valueB
 *  А?*
dtype0*
_output_shapes
: 
ч
;loss/dense_1_loss/weighted_loss/broadcast_weights/ones_likeFillAloss/dense_1_loss/weighted_loss/broadcast_weights/ones_like/ShapeAloss/dense_1_loss/weighted_loss/broadcast_weights/ones_like/Const*#
_output_shapes
:€€€€€€€€€*
T0
Ћ
1loss/dense_1_loss/weighted_loss/broadcast_weightsMul&loss/dense_1_loss/weighted_loss/Cast/x;loss/dense_1_loss/weighted_loss/broadcast_weights/ones_like*#
_output_shapes
:€€€€€€€€€*
T0
ж
#loss/dense_1_loss/weighted_loss/MulMulYloss/dense_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits1loss/dense_1_loss/weighted_loss/broadcast_weights*#
_output_shapes
:€€€€€€€€€*
T0
a
loss/dense_1_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:
{
loss/dense_1_loss/SumSum#loss/dense_1_loss/weighted_loss/Mulloss/dense_1_loss/Const*
_output_shapes
: *
T0
l
loss/dense_1_loss/num_elementsSize#loss/dense_1_loss/weighted_loss/Mul*
T0*
_output_shapes
: 
{
#loss/dense_1_loss/num_elements/CastCastloss/dense_1_loss/num_elements*
_output_shapes
: *

DstT0*

SrcT0
\
loss/dense_1_loss/Const_1Const*
valueB *
dtype0*
_output_shapes
: 
q
loss/dense_1_loss/Sum_1Sumloss/dense_1_loss/Sumloss/dense_1_loss/Const_1*
T0*
_output_shapes
: 
В
loss/dense_1_loss/valueDivNoNanloss/dense_1_loss/Sum_1#loss/dense_1_loss/num_elements/Cast*
T0*
_output_shapes
: 
O

loss/mul/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
U
loss/mulMul
loss/mul/xloss/dense_1_loss/value*
_output_shapes
: *
T0
[
training/gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
a
training/gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  А?
x
training/gradients/FillFilltraining/gradients/Shapetraining/gradients/grad_ys_0*
T0*
_output_shapes
: 
~
$training/gradients/loss/mul_grad/MulMultraining/gradients/Fillloss/dense_1_loss/value*
_output_shapes
: *
T0
s
&training/gradients/loss/mul_grad/Mul_1Multraining/gradients/Fill
loss/mul/x*
T0*
_output_shapes
: 
Й
1training/gradients/loss/mul_grad/tuple/group_depsNoOp%^training/gradients/loss/mul_grad/Mul'^training/gradients/loss/mul_grad/Mul_1
щ
9training/gradients/loss/mul_grad/tuple/control_dependencyIdentity$training/gradients/loss/mul_grad/Mul2^training/gradients/loss/mul_grad/tuple/group_deps*
T0*7
_class-
+)loc:@training/gradients/loss/mul_grad/Mul*
_output_shapes
: 
€
;training/gradients/loss/mul_grad/tuple/control_dependency_1Identity&training/gradients/loss/mul_grad/Mul_12^training/gradients/loss/mul_grad/tuple/group_deps*
T0*9
_class/
-+loc:@training/gradients/loss/mul_grad/Mul_1*
_output_shapes
: 
x
5training/gradients/loss/dense_1_loss/value_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
z
7training/gradients/loss/dense_1_loss/value_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
В
Etraining/gradients/loss/dense_1_loss/value_grad/BroadcastGradientArgsBroadcastGradientArgs5training/gradients/loss/dense_1_loss/value_grad/Shape7training/gradients/loss/dense_1_loss/value_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
…
:training/gradients/loss/dense_1_loss/value_grad/div_no_nanDivNoNan;training/gradients/loss/mul_grad/tuple/control_dependency_1#loss/dense_1_loss/num_elements/Cast*
T0*
_output_shapes
: 
ё
3training/gradients/loss/dense_1_loss/value_grad/SumSum:training/gradients/loss/dense_1_loss/value_grad/div_no_nanEtraining/gradients/loss/dense_1_loss/value_grad/BroadcastGradientArgs*
T0*
_output_shapes
: 
ѕ
7training/gradients/loss/dense_1_loss/value_grad/ReshapeReshape3training/gradients/loss/dense_1_loss/value_grad/Sum5training/gradients/loss/dense_1_loss/value_grad/Shape*
_output_shapes
: *
T0
t
3training/gradients/loss/dense_1_loss/value_grad/NegNegloss/dense_1_loss/Sum_1*
T0*
_output_shapes
: 
√
<training/gradients/loss/dense_1_loss/value_grad/div_no_nan_1DivNoNan3training/gradients/loss/dense_1_loss/value_grad/Neg#loss/dense_1_loss/num_elements/Cast*
T0*
_output_shapes
: 
ћ
<training/gradients/loss/dense_1_loss/value_grad/div_no_nan_2DivNoNan<training/gradients/loss/dense_1_loss/value_grad/div_no_nan_1#loss/dense_1_loss/num_elements/Cast*
T0*
_output_shapes
: 
÷
3training/gradients/loss/dense_1_loss/value_grad/mulMul;training/gradients/loss/mul_grad/tuple/control_dependency_1<training/gradients/loss/dense_1_loss/value_grad/div_no_nan_2*
T0*
_output_shapes
: 
џ
5training/gradients/loss/dense_1_loss/value_grad/Sum_1Sum3training/gradients/loss/dense_1_loss/value_grad/mulGtraining/gradients/loss/dense_1_loss/value_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
: 
’
9training/gradients/loss/dense_1_loss/value_grad/Reshape_1Reshape5training/gradients/loss/dense_1_loss/value_grad/Sum_17training/gradients/loss/dense_1_loss/value_grad/Shape_1*
T0*
_output_shapes
: 
Њ
@training/gradients/loss/dense_1_loss/value_grad/tuple/group_depsNoOp8^training/gradients/loss/dense_1_loss/value_grad/Reshape:^training/gradients/loss/dense_1_loss/value_grad/Reshape_1
љ
Htraining/gradients/loss/dense_1_loss/value_grad/tuple/control_dependencyIdentity7training/gradients/loss/dense_1_loss/value_grad/ReshapeA^training/gradients/loss/dense_1_loss/value_grad/tuple/group_deps*
_output_shapes
: *
T0*J
_class@
><loc:@training/gradients/loss/dense_1_loss/value_grad/Reshape
√
Jtraining/gradients/loss/dense_1_loss/value_grad/tuple/control_dependency_1Identity9training/gradients/loss/dense_1_loss/value_grad/Reshape_1A^training/gradients/loss/dense_1_loss/value_grad/tuple/group_deps*
_output_shapes
: *
T0*L
_classB
@>loc:@training/gradients/loss/dense_1_loss/value_grad/Reshape_1
А
=training/gradients/loss/dense_1_loss/Sum_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 
м
7training/gradients/loss/dense_1_loss/Sum_1_grad/ReshapeReshapeHtraining/gradients/loss/dense_1_loss/value_grad/tuple/control_dependency=training/gradients/loss/dense_1_loss/Sum_1_grad/Reshape/shape*
_output_shapes
: *
T0
x
5training/gradients/loss/dense_1_loss/Sum_1_grad/ConstConst*
valueB *
dtype0*
_output_shapes
: 
Ќ
4training/gradients/loss/dense_1_loss/Sum_1_grad/TileTile7training/gradients/loss/dense_1_loss/Sum_1_grad/Reshape5training/gradients/loss/dense_1_loss/Sum_1_grad/Const*
T0*
_output_shapes
: 
Е
;training/gradients/loss/dense_1_loss/Sum_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
Ў
5training/gradients/loss/dense_1_loss/Sum_grad/ReshapeReshape4training/gradients/loss/dense_1_loss/Sum_1_grad/Tile;training/gradients/loss/dense_1_loss/Sum_grad/Reshape/shape*
T0*
_output_shapes
:
Ж
3training/gradients/loss/dense_1_loss/Sum_grad/ShapeShape#loss/dense_1_loss/weighted_loss/Mul*
_output_shapes
:*
T0
‘
2training/gradients/loss/dense_1_loss/Sum_grad/TileTile5training/gradients/loss/dense_1_loss/Sum_grad/Reshape3training/gradients/loss/dense_1_loss/Sum_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€
 
Atraining/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/ShapeShapeYloss/dense_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*
_output_shapes
:
§
Ctraining/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/Shape_1Shape1loss/dense_1_loss/weighted_loss/broadcast_weights*
T0*
_output_shapes
:
¶
Qtraining/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/BroadcastGradientArgsBroadcastGradientArgsAtraining/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/ShapeCtraining/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
џ
?training/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/MulMul2training/gradients/loss/dense_1_loss/Sum_grad/Tile1loss/dense_1_loss/weighted_loss/broadcast_weights*
T0*#
_output_shapes
:€€€€€€€€€
э
?training/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/SumSum?training/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/MulQtraining/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:
А
Ctraining/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/ReshapeReshape?training/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/SumAtraining/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/Shape*
T0*#
_output_shapes
:€€€€€€€€€
Е
Atraining/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/Mul_1MulYloss/dense_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits2training/gradients/loss/dense_1_loss/Sum_grad/Tile*
T0*#
_output_shapes
:€€€€€€€€€
Г
Atraining/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/Sum_1SumAtraining/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/Mul_1Straining/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:
Ж
Etraining/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/Reshape_1ReshapeAtraining/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/Sum_1Ctraining/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/Shape_1*#
_output_shapes
:€€€€€€€€€*
T0
в
Ltraining/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/tuple/group_depsNoOpD^training/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/ReshapeF^training/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/Reshape_1
ъ
Ttraining/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/tuple/control_dependencyIdentityCtraining/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/ReshapeM^training/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/tuple/group_deps*
T0*V
_classL
JHloc:@training/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/Reshape*#
_output_shapes
:€€€€€€€€€
А
Vtraining/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/tuple/control_dependency_1IdentityEtraining/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/Reshape_1M^training/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/tuple/group_deps*
T0*X
_classN
LJloc:@training/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/Reshape_1*#
_output_shapes
:€€€€€€€€€
¬
training/gradients/zeros_like	ZerosLike[loss/dense_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
д
Бtraining/gradients/loss/dense_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradientPreventGradient[loss/dense_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*і
message®•Currently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()
ћ
Аtraining/gradients/loss/dense_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
Х
|training/gradients/loss/dense_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDimsTtraining/gradients/loss/dense_1_loss/weighted_loss/Mul_grad/tuple/control_dependencyАtraining/gradients/loss/dense_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*
T0*'
_output_shapes
:€€€€€€€€€
є
utraining/gradients/loss/dense_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMul|training/gradients/loss/dense_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDimsБtraining/gradients/loss/dense_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradient*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
x
9training/gradients/loss/dense_1_loss/Reshape_1_grad/ShapeShapedense_1/BiasAdd*
T0*
_output_shapes
:
™
;training/gradients/loss/dense_1_loss/Reshape_1_grad/ReshapeReshapeutraining/gradients/loss/dense_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul9training/gradients/loss/dense_1_loss/Reshape_1_grad/Shape*
T0*'
_output_shapes
:€€€€€€€€€
§
3training/gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad;training/gradients/loss/dense_1_loss/Reshape_1_grad/Reshape*
_output_shapes
:*
T0
і
8training/gradients/dense_1/BiasAdd_grad/tuple/group_depsNoOp4^training/gradients/dense_1/BiasAdd_grad/BiasAddGrad<^training/gradients/loss/dense_1_loss/Reshape_1_grad/Reshape
∆
@training/gradients/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity;training/gradients/loss/dense_1_loss/Reshape_1_grad/Reshape9^training/gradients/dense_1/BiasAdd_grad/tuple/group_deps*
T0*N
_classD
B@loc:@training/gradients/loss/dense_1_loss/Reshape_1_grad/Reshape*'
_output_shapes
:€€€€€€€€€
Ђ
Btraining/gradients/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity3training/gradients/dense_1/BiasAdd_grad/BiasAddGrad9^training/gradients/dense_1/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@training/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
Ё
-training/gradients/dense_1/MatMul_grad/MatMulMatMul@training/gradients/dense_1/BiasAdd_grad/tuple/control_dependencydense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€@*
transpose_b(
√
/training/gradients/dense_1/MatMul_grad/MatMul_1MatMul
dense/Relu@training/gradients/dense_1/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes

:@*
transpose_a(
°
7training/gradients/dense_1/MatMul_grad/tuple/group_depsNoOp.^training/gradients/dense_1/MatMul_grad/MatMul0^training/gradients/dense_1/MatMul_grad/MatMul_1
®
?training/gradients/dense_1/MatMul_grad/tuple/control_dependencyIdentity-training/gradients/dense_1/MatMul_grad/MatMul8^training/gradients/dense_1/MatMul_grad/tuple/group_deps*
T0*@
_class6
42loc:@training/gradients/dense_1/MatMul_grad/MatMul*'
_output_shapes
:€€€€€€€€€@
•
Atraining/gradients/dense_1/MatMul_grad/tuple/control_dependency_1Identity/training/gradients/dense_1/MatMul_grad/MatMul_18^training/gradients/dense_1/MatMul_grad/tuple/group_deps*
T0*B
_class8
64loc:@training/gradients/dense_1/MatMul_grad/MatMul_1*
_output_shapes

:@
ґ
+training/gradients/dense/Relu_grad/ReluGradReluGrad?training/gradients/dense_1/MatMul_grad/tuple/control_dependency
dense/Relu*
T0*'
_output_shapes
:€€€€€€€€€@
Т
1training/gradients/dense/BiasAdd_grad/BiasAddGradBiasAddGrad+training/gradients/dense/Relu_grad/ReluGrad*
T0*
_output_shapes
:@
†
6training/gradients/dense/BiasAdd_grad/tuple/group_depsNoOp2^training/gradients/dense/BiasAdd_grad/BiasAddGrad,^training/gradients/dense/Relu_grad/ReluGrad
Ґ
>training/gradients/dense/BiasAdd_grad/tuple/control_dependencyIdentity+training/gradients/dense/Relu_grad/ReluGrad7^training/gradients/dense/BiasAdd_grad/tuple/group_deps*
T0*>
_class4
20loc:@training/gradients/dense/Relu_grad/ReluGrad*'
_output_shapes
:€€€€€€€€€@
£
@training/gradients/dense/BiasAdd_grad/tuple/control_dependency_1Identity1training/gradients/dense/BiasAdd_grad/BiasAddGrad7^training/gradients/dense/BiasAdd_grad/tuple/group_deps*
T0*D
_class:
86loc:@training/gradients/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
Ў
+training/gradients/dense/MatMul_grad/MatMulMatMul>training/gradients/dense/BiasAdd_grad/tuple/control_dependencydense/MatMul/ReadVariableOp*
T0*(
_output_shapes
:€€€€€€€€€А*
transpose_b(
Ѕ
-training/gradients/dense/MatMul_grad/MatMul_1MatMuldense_input>training/gradients/dense/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	А@*
transpose_a(*
T0
Ы
5training/gradients/dense/MatMul_grad/tuple/group_depsNoOp,^training/gradients/dense/MatMul_grad/MatMul.^training/gradients/dense/MatMul_grad/MatMul_1
°
=training/gradients/dense/MatMul_grad/tuple/control_dependencyIdentity+training/gradients/dense/MatMul_grad/MatMul6^training/gradients/dense/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@training/gradients/dense/MatMul_grad/MatMul*(
_output_shapes
:€€€€€€€€€А
Ю
?training/gradients/dense/MatMul_grad/tuple/control_dependency_1Identity-training/gradients/dense/MatMul_grad/MatMul_16^training/gradients/dense/MatMul_grad/tuple/group_deps*
_output_shapes
:	А@*
T0*@
_class6
42loc:@training/gradients/dense/MatMul_grad/MatMul_1
Т
.training/beta1_power/Initializer/initial_valueConst*
_class
loc:@dense/bias*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Ы
training/beta1_powerVarHandleOp*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: *
shape: *%
shared_nametraining/beta1_power
Ш
5training/beta1_power/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/beta1_power*
_output_shapes
: *
_class
loc:@dense/bias
В
training/beta1_power/AssignAssignVariableOptraining/beta1_power.training/beta1_power/Initializer/initial_value*
dtype0
Ф
(training/beta1_power/Read/ReadVariableOpReadVariableOptraining/beta1_power*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 
Т
.training/beta2_power/Initializer/initial_valueConst*
_class
loc:@dense/bias*
valueB
 *wЊ?*
dtype0*
_output_shapes
: 
Ы
training/beta2_powerVarHandleOp*%
shared_nametraining/beta2_power*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: *
shape: 
Ш
5training/beta2_power/IsInitialized/VarIsInitializedOpVarIsInitializedOptraining/beta2_power*
_class
loc:@dense/bias*
_output_shapes
: 
В
training/beta2_power/AssignAssignVariableOptraining/beta2_power.training/beta2_power/Initializer/initial_value*
dtype0
Ф
(training/beta2_power/Read/ReadVariableOpReadVariableOptraining/beta2_power*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 
•
3dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
_class
loc:@dense/kernel*
valueB"   @   *
dtype0*
_output_shapes
:
П
)dense/kernel/Adam/Initializer/zeros/ConstConst*
_class
loc:@dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
÷
#dense/kernel/Adam/Initializer/zerosFill3dense/kernel/Adam/Initializer/zeros/shape_as_tensor)dense/kernel/Adam/Initializer/zeros/Const*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	А@
†
dense/kernel/AdamVarHandleOp*
dtype0*
_output_shapes
: *
shape:	А@*"
shared_namedense/kernel/Adam*
_class
loc:@dense/kernel
Ф
2dense/kernel/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense/kernel/Adam*
_class
loc:@dense/kernel*
_output_shapes
: 
q
dense/kernel/Adam/AssignAssignVariableOpdense/kernel/Adam#dense/kernel/Adam/Initializer/zeros*
dtype0
Щ
%dense/kernel/Adam/Read/ReadVariableOpReadVariableOpdense/kernel/Adam*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:	А@
І
5dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
_class
loc:@dense/kernel*
valueB"   @   
С
+dense/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
_class
loc:@dense/kernel*
valueB
 *    
№
%dense/kernel/Adam_1/Initializer/zerosFill5dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor+dense/kernel/Adam_1/Initializer/zeros/Const*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	А@
§
dense/kernel/Adam_1VarHandleOp*
dtype0*
_output_shapes
: *
shape:	А@*$
shared_namedense/kernel/Adam_1*
_class
loc:@dense/kernel
Ш
4dense/kernel/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense/kernel/Adam_1*
_class
loc:@dense/kernel*
_output_shapes
: 
w
dense/kernel/Adam_1/AssignAssignVariableOpdense/kernel/Adam_1%dense/kernel/Adam_1/Initializer/zeros*
dtype0
Э
'dense/kernel/Adam_1/Read/ReadVariableOpReadVariableOpdense/kernel/Adam_1*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:	А@
Н
!dense/bias/Adam/Initializer/zerosConst*
_class
loc:@dense/bias*
valueB@*    *
dtype0*
_output_shapes
:@
Х
dense/bias/AdamVarHandleOp*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: *
shape:@* 
shared_namedense/bias/Adam
О
0dense/bias/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense/bias/Adam*
_class
loc:@dense/bias*
_output_shapes
: 
k
dense/bias/Adam/AssignAssignVariableOpdense/bias/Adam!dense/bias/Adam/Initializer/zeros*
dtype0
О
#dense/bias/Adam/Read/ReadVariableOpReadVariableOpdense/bias/Adam*
_class
loc:@dense/bias*
dtype0*
_output_shapes
:@
П
#dense/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:@*
_class
loc:@dense/bias*
valueB@*    
Щ
dense/bias/Adam_1VarHandleOp*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: *
shape:@*"
shared_namedense/bias/Adam_1
Т
2dense/bias/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense/bias/Adam_1*
_class
loc:@dense/bias*
_output_shapes
: 
q
dense/bias/Adam_1/AssignAssignVariableOpdense/bias/Adam_1#dense/bias/Adam_1/Initializer/zeros*
dtype0
Т
%dense/bias/Adam_1/Read/ReadVariableOpReadVariableOpdense/bias/Adam_1*
dtype0*
_output_shapes
:@*
_class
loc:@dense/bias
©
5dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*!
_class
loc:@dense_1/kernel*
valueB"@      *
dtype0*
_output_shapes
:
У
+dense_1/kernel/Adam/Initializer/zeros/ConstConst*!
_class
loc:@dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Ё
%dense_1/kernel/Adam/Initializer/zerosFill5dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor+dense_1/kernel/Adam/Initializer/zeros/Const*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:@
•
dense_1/kernel/AdamVarHandleOp*
shape
:@*$
shared_namedense_1/kernel/Adam*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
Ъ
4dense_1/kernel/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/kernel/Adam*!
_class
loc:@dense_1/kernel*
_output_shapes
: 
w
dense_1/kernel/Adam/AssignAssignVariableOpdense_1/kernel/Adam%dense_1/kernel/Adam/Initializer/zeros*
dtype0
Ю
'dense_1/kernel/Adam/Read/ReadVariableOpReadVariableOpdense_1/kernel/Adam*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes

:@
Ђ
7dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*!
_class
loc:@dense_1/kernel*
valueB"@      *
dtype0*
_output_shapes
:
Х
-dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*!
_class
loc:@dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
г
'dense_1/kernel/Adam_1/Initializer/zerosFill7dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor-dense_1/kernel/Adam_1/Initializer/zeros/Const*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:@
©
dense_1/kernel/Adam_1VarHandleOp*
dtype0*
_output_shapes
: *
shape
:@*&
shared_namedense_1/kernel/Adam_1*!
_class
loc:@dense_1/kernel
Ю
6dense_1/kernel/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/kernel/Adam_1*!
_class
loc:@dense_1/kernel*
_output_shapes
: 
}
dense_1/kernel/Adam_1/AssignAssignVariableOpdense_1/kernel/Adam_1'dense_1/kernel/Adam_1/Initializer/zeros*
dtype0
Ґ
)dense_1/kernel/Adam_1/Read/ReadVariableOpReadVariableOpdense_1/kernel/Adam_1*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes

:@
С
#dense_1/bias/Adam/Initializer/zerosConst*
_class
loc:@dense_1/bias*
valueB*    *
dtype0*
_output_shapes
:
Ы
dense_1/bias/AdamVarHandleOp*
dtype0*
_output_shapes
: *
shape:*"
shared_namedense_1/bias/Adam*
_class
loc:@dense_1/bias
Ф
2dense_1/bias/Adam/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/bias/Adam*
_output_shapes
: *
_class
loc:@dense_1/bias
q
dense_1/bias/Adam/AssignAssignVariableOpdense_1/bias/Adam#dense_1/bias/Adam/Initializer/zeros*
dtype0
Ф
%dense_1/bias/Adam/Read/ReadVariableOpReadVariableOpdense_1/bias/Adam*
dtype0*
_output_shapes
:*
_class
loc:@dense_1/bias
У
%dense_1/bias/Adam_1/Initializer/zerosConst*
_class
loc:@dense_1/bias*
valueB*    *
dtype0*
_output_shapes
:
Я
dense_1/bias/Adam_1VarHandleOp*$
shared_namedense_1/bias/Adam_1*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: *
shape:
Ш
4dense_1/bias/Adam_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/bias/Adam_1*
_class
loc:@dense_1/bias*
_output_shapes
: 
w
dense_1/bias/Adam_1/AssignAssignVariableOpdense_1/bias/Adam_1%dense_1/bias/Adam_1/Initializer/zeros*
dtype0
Ш
'dense_1/bias/Adam_1/Read/ReadVariableOpReadVariableOpdense_1/bias/Adam_1*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
:
`
training/Adam/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *oГ:
X
training/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
X
training/Adam/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *wЊ?
Z
training/Adam/epsilonConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
П
Btraining/Adam/update_dense/kernel/ResourceApplyAdam/ReadVariableOpReadVariableOptraining/beta1_power*
dtype0*
_output_shapes
: 
С
Dtraining/Adam/update_dense/kernel/ResourceApplyAdam/ReadVariableOp_1ReadVariableOptraining/beta2_power*
dtype0*
_output_shapes
: 
—
3training/Adam/update_dense/kernel/ResourceApplyAdamResourceApplyAdamdense/kerneldense/kernel/Adamdense/kernel/Adam_1Btraining/Adam/update_dense/kernel/ResourceApplyAdam/ReadVariableOpDtraining/Adam/update_dense/kernel/ResourceApplyAdam/ReadVariableOp_1training/Adam/learning_ratetraining/Adam/beta1training/Adam/beta2training/Adam/epsilon?training/gradients/dense/MatMul_grad/tuple/control_dependency_1*
T0*
_class
loc:@dense/kernel
Н
@training/Adam/update_dense/bias/ResourceApplyAdam/ReadVariableOpReadVariableOptraining/beta1_power*
dtype0*
_output_shapes
: 
П
Btraining/Adam/update_dense/bias/ResourceApplyAdam/ReadVariableOp_1ReadVariableOptraining/beta2_power*
dtype0*
_output_shapes
: 
ƒ
1training/Adam/update_dense/bias/ResourceApplyAdamResourceApplyAdam
dense/biasdense/bias/Adamdense/bias/Adam_1@training/Adam/update_dense/bias/ResourceApplyAdam/ReadVariableOpBtraining/Adam/update_dense/bias/ResourceApplyAdam/ReadVariableOp_1training/Adam/learning_ratetraining/Adam/beta1training/Adam/beta2training/Adam/epsilon@training/gradients/dense/BiasAdd_grad/tuple/control_dependency_1*
T0*
_class
loc:@dense/bias
С
Dtraining/Adam/update_dense_1/kernel/ResourceApplyAdam/ReadVariableOpReadVariableOptraining/beta1_power*
dtype0*
_output_shapes
: 
У
Ftraining/Adam/update_dense_1/kernel/ResourceApplyAdam/ReadVariableOp_1ReadVariableOptraining/beta2_power*
dtype0*
_output_shapes
: 
б
5training/Adam/update_dense_1/kernel/ResourceApplyAdamResourceApplyAdamdense_1/kerneldense_1/kernel/Adamdense_1/kernel/Adam_1Dtraining/Adam/update_dense_1/kernel/ResourceApplyAdam/ReadVariableOpFtraining/Adam/update_dense_1/kernel/ResourceApplyAdam/ReadVariableOp_1training/Adam/learning_ratetraining/Adam/beta1training/Adam/beta2training/Adam/epsilonAtraining/gradients/dense_1/MatMul_grad/tuple/control_dependency_1*
T0*!
_class
loc:@dense_1/kernel
П
Btraining/Adam/update_dense_1/bias/ResourceApplyAdam/ReadVariableOpReadVariableOptraining/beta1_power*
dtype0*
_output_shapes
: 
С
Dtraining/Adam/update_dense_1/bias/ResourceApplyAdam/ReadVariableOp_1ReadVariableOptraining/beta2_power*
dtype0*
_output_shapes
: 
‘
3training/Adam/update_dense_1/bias/ResourceApplyAdamResourceApplyAdamdense_1/biasdense_1/bias/Adamdense_1/bias/Adam_1Btraining/Adam/update_dense_1/bias/ResourceApplyAdam/ReadVariableOpDtraining/Adam/update_dense_1/bias/ResourceApplyAdam/ReadVariableOp_1training/Adam/learning_ratetraining/Adam/beta1training/Adam/beta2training/Adam/epsilonBtraining/gradients/dense_1/BiasAdd_grad/tuple/control_dependency_1*
T0*
_class
loc:@dense_1/bias
Ѕ
training/Adam/ReadVariableOpReadVariableOptraining/beta1_power2^training/Adam/update_dense/bias/ResourceApplyAdam4^training/Adam/update_dense/kernel/ResourceApplyAdam4^training/Adam/update_dense_1/bias/ResourceApplyAdam6^training/Adam/update_dense_1/kernel/ResourceApplyAdam*
dtype0*
_output_shapes
: 
Л
training/Adam/mulMultraining/Adam/ReadVariableOptraining/Adam/beta1*
_output_shapes
: *
T0*
_class
loc:@dense/bias
З
training/Adam/AssignVariableOpAssignVariableOptraining/beta1_powertraining/Adam/mul*
_class
loc:@dense/bias*
dtype0
Г
training/Adam/ReadVariableOp_1ReadVariableOptraining/beta1_power^training/Adam/AssignVariableOp2^training/Adam/update_dense/bias/ResourceApplyAdam4^training/Adam/update_dense/kernel/ResourceApplyAdam4^training/Adam/update_dense_1/bias/ResourceApplyAdam6^training/Adam/update_dense_1/kernel/ResourceApplyAdam*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 
√
training/Adam/ReadVariableOp_2ReadVariableOptraining/beta2_power2^training/Adam/update_dense/bias/ResourceApplyAdam4^training/Adam/update_dense/kernel/ResourceApplyAdam4^training/Adam/update_dense_1/bias/ResourceApplyAdam6^training/Adam/update_dense_1/kernel/ResourceApplyAdam*
dtype0*
_output_shapes
: 
П
training/Adam/mul_1Multraining/Adam/ReadVariableOp_2training/Adam/beta2*
T0*
_class
loc:@dense/bias*
_output_shapes
: 
Л
 training/Adam/AssignVariableOp_1AssignVariableOptraining/beta2_powertraining/Adam/mul_1*
dtype0*
_class
loc:@dense/bias
Е
training/Adam/ReadVariableOp_3ReadVariableOptraining/beta2_power!^training/Adam/AssignVariableOp_12^training/Adam/update_dense/bias/ResourceApplyAdam4^training/Adam/update_dense/kernel/ResourceApplyAdam4^training/Adam/update_dense_1/bias/ResourceApplyAdam6^training/Adam/update_dense_1/kernel/ResourceApplyAdam*
dtype0*
_output_shapes
: *
_class
loc:@dense/bias
Є
training/Adam/updateNoOp^training/Adam/AssignVariableOp!^training/Adam/AssignVariableOp_12^training/Adam/update_dense/bias/ResourceApplyAdam4^training/Adam/update_dense/kernel/ResourceApplyAdam4^training/Adam/update_dense_1/bias/ResourceApplyAdam6^training/Adam/update_dense_1/kernel/ResourceApplyAdam
Ч
training/Adam/ConstConst^training/Adam/update*
value	B	 R*)
_class
loc:@TFOptimizer/iterations*
dtype0	*
_output_shapes
: 
Й
training/AdamAssignAddVariableOpTFOptimizer/iterationstraining/Adam/Const*)
_class
loc:@TFOptimizer/iterations*
dtype0	
8
training_1/group_depsNoOp	^loss/mul^training/Adam
Z
ConstConst"/device:CPU:0*
dtype0*
_output_shapes
: *
valueB Bmodel
 
RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*q
valuehBfB\layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
r
RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
Л
	RestoreV2	RestoreV2ConstRestoreV2/tensor_namesRestoreV2/shape_and_slices"/device:CPU:0*
_output_shapes
:*
dtypes
2
B
IdentityIdentity	RestoreV2*
T0*
_output_shapes
:
N
AssignVariableOpAssignVariableOpdense/kernel/AdamIdentity*
dtype0
ћ
RestoreV2_1/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*q
valuehBfB\layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
С
RestoreV2_1	RestoreV2ConstRestoreV2_1/tensor_namesRestoreV2_1/shape_and_slices"/device:CPU:0*
_output_shapes
:*
dtypes
2
F

Identity_1IdentityRestoreV2_1*
T0*
_output_shapes
:
T
AssignVariableOp_1AssignVariableOpdense/kernel/Adam_1
Identity_1*
dtype0
 
RestoreV2_2/tensor_namesConst"/device:CPU:0*o
valuefBdBZlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_2/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
С
RestoreV2_2	RestoreV2ConstRestoreV2_2/tensor_namesRestoreV2_2/shape_and_slices"/device:CPU:0*
_output_shapes
:*
dtypes
2
F

Identity_2IdentityRestoreV2_2*
T0*
_output_shapes
:
P
AssignVariableOp_2AssignVariableOpdense/bias/Adam
Identity_2*
dtype0
 
RestoreV2_3/tensor_namesConst"/device:CPU:0*o
valuefBdBZlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_3/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
С
RestoreV2_3	RestoreV2ConstRestoreV2_3/tensor_namesRestoreV2_3/shape_and_slices"/device:CPU:0*
_output_shapes
:*
dtypes
2
F

Identity_3IdentityRestoreV2_3*
_output_shapes
:*
T0
R
AssignVariableOp_3AssignVariableOpdense/bias/Adam_1
Identity_3*
dtype0
ћ
RestoreV2_4/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*q
valuehBfB\layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
t
RestoreV2_4/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
С
RestoreV2_4	RestoreV2ConstRestoreV2_4/tensor_namesRestoreV2_4/shape_and_slices"/device:CPU:0*
_output_shapes
:*
dtypes
2
F

Identity_4IdentityRestoreV2_4*
T0*
_output_shapes
:
T
AssignVariableOp_4AssignVariableOpdense_1/kernel/Adam
Identity_4*
dtype0
ћ
RestoreV2_5/tensor_namesConst"/device:CPU:0*q
valuehBfB\layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_5/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
С
RestoreV2_5	RestoreV2ConstRestoreV2_5/tensor_namesRestoreV2_5/shape_and_slices"/device:CPU:0*
dtypes
2*
_output_shapes
:
F

Identity_5IdentityRestoreV2_5*
_output_shapes
:*
T0
V
AssignVariableOp_5AssignVariableOpdense_1/kernel/Adam_1
Identity_5*
dtype0
 
RestoreV2_6/tensor_namesConst"/device:CPU:0*o
valuefBdBZlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_6/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
С
RestoreV2_6	RestoreV2ConstRestoreV2_6/tensor_namesRestoreV2_6/shape_and_slices"/device:CPU:0*
_output_shapes
:*
dtypes
2
F

Identity_6IdentityRestoreV2_6*
T0*
_output_shapes
:
R
AssignVariableOp_6AssignVariableOpdense_1/bias/Adam
Identity_6*
dtype0
 
RestoreV2_7/tensor_namesConst"/device:CPU:0*o
valuefBdBZlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
t
RestoreV2_7/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
С
RestoreV2_7	RestoreV2ConstRestoreV2_7/tensor_namesRestoreV2_7/shape_and_slices"/device:CPU:0*
_output_shapes
:*
dtypes
2
F

Identity_7IdentityRestoreV2_7*
T0*
_output_shapes
:
T
AssignVariableOp_7AssignVariableOpdense_1/bias/Adam_1
Identity_7*
dtype0
ч
RestoreV2_8/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*Ы
valueСBОB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/global_step/.ATTRIBUTES/VARIABLE_VALUEB:optimizer/optimizer/beta1_power/.ATTRIBUTES/VARIABLE_VALUEB:optimizer/optimizer/beta2_power/.ATTRIBUTES/VARIABLE_VALUE
А
RestoreV2_8/shape_and_slicesConst"/device:CPU:0*!
valueBB B B B B B B *
dtype0*
_output_shapes
:
ѓ
RestoreV2_8	RestoreV2ConstRestoreV2_8/tensor_namesRestoreV2_8/shape_and_slices"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	2	
F

Identity_8IdentityRestoreV2_8*
T0*
_output_shapes
:
K
AssignVariableOp_8AssignVariableOp
dense/bias
Identity_8*
dtype0
H

Identity_9IdentityRestoreV2_8:1*
T0*
_output_shapes
:
M
AssignVariableOp_9AssignVariableOpdense/kernel
Identity_9*
dtype0
I
Identity_10IdentityRestoreV2_8:2*
T0*
_output_shapes
:
O
AssignVariableOp_10AssignVariableOpdense_1/biasIdentity_10*
dtype0
I
Identity_11IdentityRestoreV2_8:3*
T0*
_output_shapes
:
Q
AssignVariableOp_11AssignVariableOpdense_1/kernelIdentity_11*
dtype0
I
Identity_12IdentityRestoreV2_8:4*
_output_shapes
:*
T0	
Y
AssignVariableOp_12AssignVariableOpTFOptimizer/iterationsIdentity_12*
dtype0	
I
Identity_13IdentityRestoreV2_8:5*
T0*
_output_shapes
:
W
AssignVariableOp_13AssignVariableOptraining/beta1_powerIdentity_13*
dtype0
I
Identity_14IdentityRestoreV2_8:6*
T0*
_output_shapes
:
W
AssignVariableOp_14AssignVariableOptraining/beta2_powerIdentity_14*
dtype0
W
VarIsInitializedOpVarIsInitializedOpdense_1/kernel/Adam_1*
_output_shapes
: 
S
VarIsInitializedOp_1VarIsInitializedOpdense/bias/Adam*
_output_shapes
: 
P
VarIsInitializedOp_2VarIsInitializedOpdense/kernel*
_output_shapes
: 
Z
VarIsInitializedOp_3VarIsInitializedOpTFOptimizer/iterations*
_output_shapes
: 
R
VarIsInitializedOp_4VarIsInitializedOpdense_1/kernel*
_output_shapes
: 
P
VarIsInitializedOp_5VarIsInitializedOpdense_1/bias*
_output_shapes
: 
I
VarIsInitializedOp_6VarIsInitializedOptotal*
_output_shapes
: 
U
VarIsInitializedOp_7VarIsInitializedOpdense/bias/Adam_1*
_output_shapes
: 
U
VarIsInitializedOp_8VarIsInitializedOpdense/kernel/Adam*
_output_shapes
: 
W
VarIsInitializedOp_9VarIsInitializedOpdense/kernel/Adam_1*
_output_shapes
: 
V
VarIsInitializedOp_10VarIsInitializedOpdense_1/bias/Adam*
_output_shapes
: 
Y
VarIsInitializedOp_11VarIsInitializedOptraining/beta1_power*
_output_shapes
: 
J
VarIsInitializedOp_12VarIsInitializedOpcount*
_output_shapes
: 
Y
VarIsInitializedOp_13VarIsInitializedOptraining/beta2_power*
_output_shapes
: 
X
VarIsInitializedOp_14VarIsInitializedOpdense_1/kernel/Adam*
_output_shapes
: 
O
VarIsInitializedOp_15VarIsInitializedOp
dense/bias*
_output_shapes
: 
X
VarIsInitializedOp_16VarIsInitializedOpdense_1/bias/Adam_1*
_output_shapes
: 
Њ
initNoOp^TFOptimizer/iterations/Assign^count/Assign^dense/bias/Adam/Assign^dense/bias/Adam_1/Assign^dense/bias/Assign^dense/kernel/Adam/Assign^dense/kernel/Adam_1/Assign^dense/kernel/Assign^dense_1/bias/Adam/Assign^dense_1/bias/Adam_1/Assign^dense_1/bias/Assign^dense_1/kernel/Adam/Assign^dense_1/kernel/Adam_1/Assign^dense_1/kernel/Assign^total/Assign^training/beta1_power/Assign^training/beta2_power/Assign
W
Const_1Const"/device:CPU:0*
dtype0*
_output_shapes
: *
valueB B 
W
Const_2Const"/device:CPU:0*
dtype0*
_output_shapes
: *
valueB B 
О
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_ff69f718569b4e2c901c4f0c3aee58cd/part*
dtype0*
_output_shapes
: 
f

StringJoin
StringJoinConst_2StringJoin/inputs_1"/device:CPU:0*
N*
_output_shapes
: 
L

num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
f
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
x
ShardedFilenameShardedFilename
StringJoinShardedFilename/shard
num_shards"/device:CPU:0*
_output_shapes
: 
Џ	
SaveV2/tensor_namesConst"/device:CPU:0*Г	
valueщBцB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/global_step/.ATTRIBUTES/VARIABLE_VALUEB:optimizer/optimizer/beta1_power/.ATTRIBUTES/VARIABLE_VALUEB:optimizer/optimizer/beta2_power/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
Л
SaveV2/shape_and_slicesConst"/device:CPU:0*1
value(B&B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
≈
SaveV2SaveV2ShardedFilenameSaveV2/tensor_namesSaveV2/shape_and_slices dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp*TFOptimizer/iterations/Read/ReadVariableOp(training/beta1_power/Read/ReadVariableOp(training/beta2_power/Read/ReadVariableOp%dense/kernel/Adam/Read/ReadVariableOp#dense/bias/Adam/Read/ReadVariableOp'dense_1/kernel/Adam/Read/ReadVariableOp%dense_1/bias/Adam/Read/ReadVariableOp'dense/kernel/Adam_1/Read/ReadVariableOp%dense/bias/Adam_1/Read/ReadVariableOp)dense_1/kernel/Adam_1/Read/ReadVariableOp'dense_1/bias/Adam_1/Read/ReadVariableOp"/device:CPU:0*
dtypes
2	
h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: 
|
ShardedFilename_1ShardedFilename
StringJoinShardedFilename_1/shard
num_shards"/device:CPU:0*
_output_shapes
: 
Й
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:
q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB
B 
А
SaveV2_1SaveV2ShardedFilename_1SaveV2_1/tensor_namesSaveV2_1/shape_and_slicesConst_1"/device:CPU:0*
dtypes
2
£
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilenameShardedFilename_1^SaveV2	^SaveV2_1"/device:CPU:0*
N*
_output_shapes
:*
T0
h
MergeV2CheckpointsMergeV2Checkpoints&MergeV2Checkpoints/checkpoint_prefixesConst_2"/device:CPU:0
e
Identity_15IdentityConst_2^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 
W
div_no_nan/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
Y
div_no_nan/ReadVariableOp_1ReadVariableOpcount*
dtype0*
_output_shapes
: 
o

div_no_nanDivNoNandiv_no_nan/ReadVariableOpdiv_no_nan/ReadVariableOp_1*
_output_shapes
: *
T0
D
Identity_16Identity
div_no_nan*
T0*
_output_shapes
: 
x
metric_op_wrapperConst"^metrics/acc/AssignAddVariableOp_1*
valueB *
dtype0*
_output_shapes
: 
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
_output_shapes
: *
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 
–	
save/SaveV2/tensor_namesConst*Г	
valueщBцB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/global_step/.ATTRIBUTES/VARIABLE_VALUEB:optimizer/optimizer/beta1_power/.ATTRIBUTES/VARIABLE_VALUEB:optimizer/optimizer/beta2_power/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
Б
save/SaveV2/shape_and_slicesConst*1
value(B&B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ј
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesdense/bias/Read/ReadVariableOp#dense/bias/Adam/Read/ReadVariableOp%dense/bias/Adam_1/Read/ReadVariableOp dense/kernel/Read/ReadVariableOp%dense/kernel/Adam/Read/ReadVariableOp'dense/kernel/Adam_1/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp%dense_1/bias/Adam/Read/ReadVariableOp'dense_1/bias/Adam_1/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp'dense_1/kernel/Adam/Read/ReadVariableOp)dense_1/kernel/Adam_1/Read/ReadVariableOp*TFOptimizer/iterations/Read/ReadVariableOp(training/beta1_power/Read/ReadVariableOp(training/beta2_power/Read/ReadVariableOp*
dtypes
2	
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
в	
save/RestoreV2/tensor_namesConst"/device:CPU:0*Г	
valueщBцB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/global_step/.ATTRIBUTES/VARIABLE_VALUEB:optimizer/optimizer/beta1_power/.ATTRIBUTES/VARIABLE_VALUEB:optimizer/optimizer/beta2_power/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
У
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*1
value(B&B B B B B B B B B B B B B B B 
е
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2	*P
_output_shapes>
<:::::::::::::::
L
save/IdentityIdentitysave/RestoreV2*
T0*
_output_shapes
:
Q
save/AssignVariableOpAssignVariableOp
dense/biassave/Identity*
dtype0
P
save/Identity_1Identitysave/RestoreV2:1*
_output_shapes
:*
T0
Z
save/AssignVariableOp_1AssignVariableOpdense/bias/Adamsave/Identity_1*
dtype0
P
save/Identity_2Identitysave/RestoreV2:2*
_output_shapes
:*
T0
\
save/AssignVariableOp_2AssignVariableOpdense/bias/Adam_1save/Identity_2*
dtype0
P
save/Identity_3Identitysave/RestoreV2:3*
T0*
_output_shapes
:
W
save/AssignVariableOp_3AssignVariableOpdense/kernelsave/Identity_3*
dtype0
P
save/Identity_4Identitysave/RestoreV2:4*
T0*
_output_shapes
:
\
save/AssignVariableOp_4AssignVariableOpdense/kernel/Adamsave/Identity_4*
dtype0
P
save/Identity_5Identitysave/RestoreV2:5*
T0*
_output_shapes
:
^
save/AssignVariableOp_5AssignVariableOpdense/kernel/Adam_1save/Identity_5*
dtype0
P
save/Identity_6Identitysave/RestoreV2:6*
T0*
_output_shapes
:
W
save/AssignVariableOp_6AssignVariableOpdense_1/biassave/Identity_6*
dtype0
P
save/Identity_7Identitysave/RestoreV2:7*
T0*
_output_shapes
:
\
save/AssignVariableOp_7AssignVariableOpdense_1/bias/Adamsave/Identity_7*
dtype0
P
save/Identity_8Identitysave/RestoreV2:8*
_output_shapes
:*
T0
^
save/AssignVariableOp_8AssignVariableOpdense_1/bias/Adam_1save/Identity_8*
dtype0
P
save/Identity_9Identitysave/RestoreV2:9*
T0*
_output_shapes
:
Y
save/AssignVariableOp_9AssignVariableOpdense_1/kernelsave/Identity_9*
dtype0
R
save/Identity_10Identitysave/RestoreV2:10*
T0*
_output_shapes
:
`
save/AssignVariableOp_10AssignVariableOpdense_1/kernel/Adamsave/Identity_10*
dtype0
R
save/Identity_11Identitysave/RestoreV2:11*
T0*
_output_shapes
:
b
save/AssignVariableOp_11AssignVariableOpdense_1/kernel/Adam_1save/Identity_11*
dtype0
R
save/Identity_12Identitysave/RestoreV2:12*
_output_shapes
:*
T0	
c
save/AssignVariableOp_12AssignVariableOpTFOptimizer/iterationssave/Identity_12*
dtype0	
R
save/Identity_13Identitysave/RestoreV2:13*
T0*
_output_shapes
:
a
save/AssignVariableOp_13AssignVariableOptraining/beta1_powersave/Identity_13*
dtype0
R
save/Identity_14Identitysave/RestoreV2:14*
T0*
_output_shapes
:
a
save/AssignVariableOp_14AssignVariableOptraining/beta2_powersave/Identity_14*
dtype0
°
save/restore_allNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_10^save/AssignVariableOp_11^save/AssignVariableOp_12^save/AssignVariableOp_13^save/AssignVariableOp_14^save/AssignVariableOp_2^save/AssignVariableOp_3^save/AssignVariableOp_4^save/AssignVariableOp_5^save/AssignVariableOp_6^save/AssignVariableOp_7^save/AssignVariableOp_8^save/AssignVariableOp_9
,
init_1NoOp^count/Assign^total/Assign"ЖD
save/Const:0save/control_dependency:0save/restore_all 5 @F8"µ
global_step•Ґ
Я
TFOptimizer/iterations:0TFOptimizer/iterations/Assign,TFOptimizer/iterations/Read/ReadVariableOp:0(22TFOptimizer/iterations/Initializer/initial_value:08"
train_op

training/Adam"щ
	variablesли
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08
А
dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2+dense_1/kernel/Initializer/random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08
Я
TFOptimizer/iterations:0TFOptimizer/iterations/Assign,TFOptimizer/iterations/Read/ReadVariableOp:0(22TFOptimizer/iterations/Initializer/initial_value:08
Х
training/beta1_power:0training/beta1_power/Assign*training/beta1_power/Read/ReadVariableOp:0(20training/beta1_power/Initializer/initial_value:0
Х
training/beta2_power:0training/beta2_power/Assign*training/beta2_power/Read/ReadVariableOp:0(20training/beta2_power/Initializer/initial_value:0
Б
dense/kernel/Adam:0dense/kernel/Adam/Assign'dense/kernel/Adam/Read/ReadVariableOp:0(2%dense/kernel/Adam/Initializer/zeros:0
Й
dense/kernel/Adam_1:0dense/kernel/Adam_1/Assign)dense/kernel/Adam_1/Read/ReadVariableOp:0(2'dense/kernel/Adam_1/Initializer/zeros:0
y
dense/bias/Adam:0dense/bias/Adam/Assign%dense/bias/Adam/Read/ReadVariableOp:0(2#dense/bias/Adam/Initializer/zeros:0
Б
dense/bias/Adam_1:0dense/bias/Adam_1/Assign'dense/bias/Adam_1/Read/ReadVariableOp:0(2%dense/bias/Adam_1/Initializer/zeros:0
Й
dense_1/kernel/Adam:0dense_1/kernel/Adam/Assign)dense_1/kernel/Adam/Read/ReadVariableOp:0(2'dense_1/kernel/Adam/Initializer/zeros:0
С
dense_1/kernel/Adam_1:0dense_1/kernel/Adam_1/Assign+dense_1/kernel/Adam_1/Read/ReadVariableOp:0(2)dense_1/kernel/Adam_1/Initializer/zeros:0
Б
dense_1/bias/Adam:0dense_1/bias/Adam/Assign'dense_1/bias/Adam/Read/ReadVariableOp:0(2%dense_1/bias/Adam/Initializer/zeros:0
Й
dense_1/bias/Adam_1:0dense_1/bias/Adam_1/Assign)dense_1/bias/Adam_1/Read/ReadVariableOp:0(2'dense_1/bias/Adam_1/Initializer/zeros:0"Ф
trainable_variablesьщ
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08
А
dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2+dense_1/kernel/Initializer/random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08
Я
TFOptimizer/iterations:0TFOptimizer/iterations/Assign,TFOptimizer/iterations/Read/ReadVariableOp:0(22TFOptimizer/iterations/Initializer/initial_value:08"≈
local_variables±Ѓ
U
total:0total/Assigntotal/Read/ReadVariableOp:0(2total/Initializer/zeros:0@H
U
count:0count/Assigncount/Read/ReadVariableOp:0(2count/Initializer/zeros:0@H*Q
__saved_model_train_op75
__saved_model_train_op
training_1/group_deps*я
train’
B
dense_1_target0
dense_1_target:0€€€€€€€€€€€€€€€€€€
4
dense_input%
dense_input:0€€€€€€€€€А?
predictions/dense_1(
dense_1/Softmax:0€€€€€€€€€4
metrics/acc/update_op
metric_op_wrapper:0 (
metrics/acc/value
Identity_16:0 
loss

loss/mul:0 tensorflow/supervised/training*@
__saved_model_init_op'%
__saved_model_init_op
init_1ОЋ
т∆
:
Add
x"T
y"T
z"T"
Ttype:
2	
Ы
ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
E
AssignAddVariableOp
resource
value"dtype"
dtypetypeИ
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
8
DivNoNan
x"T
y"T
z"T"
Ttype:	
2
h
Equal
x"T
y"T
z
"
Ttype:
2	
"$
incompatible_shape_errorbool(Р
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
У
#SparseSoftmaxCrossEntropyWithLogits
features"T
labels"Tlabels	
loss"T
backprop"T"
Ttype:
2"
Tlabelstype0	:
2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
ц
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
М
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ
9
VarIsInitializedOp
resource
is_initialized
И"eval*1.15.02v1.15.0-rc3-22-g590d6ee8НҐ
p
dense_inputPlaceholder*
dtype0*(
_output_shapes
:€€€€€€€€€А*
shape:€€€€€€€€€А
Я
-dense/kernel/Initializer/random_uniform/shapeConst*
valueB"   @   *
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:
С
+dense/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *тк≠љ*
_class
loc:@dense/kernel
С
+dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *тк≠=*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
Ќ
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	А@*
T0*
_class
loc:@dense/kernel
ќ
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
: 
б
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	А@
”
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
_output_shapes
:	А@*
T0*
_class
loc:@dense/kernel
Ц
dense/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:	А@*
shared_namedense/kernel*
_class
loc:@dense/kernel
i
-dense/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense/kernel*
_output_shapes
: 
k
dense/kernel/AssignAssignVariableOpdense/kernel'dense/kernel/Initializer/random_uniform*
dtype0
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes
:	А@
И
dense/bias/Initializer/zerosConst*
valueB@*    *
_class
loc:@dense/bias*
dtype0*
_output_shapes
:@
Л

dense/biasVarHandleOp*
shared_name
dense/bias*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: *
shape:@
e
+dense/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp
dense/bias*
_output_shapes
: 
\
dense/bias/AssignAssignVariableOp
dense/biasdense/bias/Initializer/zeros*
dtype0
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:@
i
dense/MatMul/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes
:	А@
r
dense/MatMulMatMuldense_inputdense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€@
c
dense/BiasAdd/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:@
v
dense/BiasAddBiasAdddense/MatMuldense/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€@
S

dense/ReluReludense/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€@
£
/dense_1/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"@      *!
_class
loc:@dense_1/kernel
Х
-dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *2uЖЊ*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
Х
-dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *2uЖ>*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
“
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
T0*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes

:@
÷
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes
: 
и
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:@
Џ
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:@
Ы
dense_1/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape
:@*
shared_namedense_1/kernel*!
_class
loc:@dense_1/kernel
m
/dense_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/kernel*
_output_shapes
: 
q
dense_1/kernel/AssignAssignVariableOpdense_1/kernel)dense_1/kernel/Initializer/random_uniform*
dtype0
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes

:@
М
dense_1/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *
_class
loc:@dense_1/bias
С
dense_1/biasVarHandleOp*
shared_namedense_1/bias*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: *
shape:
i
-dense_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/bias*
_output_shapes
: 
b
dense_1/bias/AssignAssignVariableOpdense_1/biasdense_1/bias/Initializer/zeros*
dtype0
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes
:
l
dense_1/MatMul/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes

:@
u
dense_1/MatMulMatMul
dense/Reludense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€
g
dense_1/BiasAdd/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes
:
|
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€
]
dense_1/SoftmaxSoftmaxdense_1/BiasAdd*'
_output_shapes
:€€€€€€€€€*
T0
Э
0TFOptimizer/iterations/Initializer/initial_valueConst*
dtype0	*
_output_shapes
: *
value	B	 R *)
_class
loc:@TFOptimizer/iterations
Ђ
TFOptimizer/iterationsVarHandleOp*
shape: *'
shared_nameTFOptimizer/iterations*)
_class
loc:@TFOptimizer/iterations*
dtype0	*
_output_shapes
: 
}
7TFOptimizer/iterations/IsInitialized/VarIsInitializedOpVarIsInitializedOpTFOptimizer/iterations*
_output_shapes
: 
И
TFOptimizer/iterations/AssignAssignVariableOpTFOptimizer/iterations0TFOptimizer/iterations/Initializer/initial_value*
dtype0	
y
*TFOptimizer/iterations/Read/ReadVariableOpReadVariableOpTFOptimizer/iterations*
dtype0	*
_output_shapes
: 
Г
dense_1_targetPlaceholder*%
shape:€€€€€€€€€€€€€€€€€€*
dtype0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
v
total/Initializer/zerosConst*
valueB
 *    *
_class

loc:@total*
dtype0*
_output_shapes
: 
x
totalVarHandleOp*
shared_nametotal*
_class

loc:@total*
dtype0*
_output_shapes
: *
shape: 
[
&total/IsInitialized/VarIsInitializedOpVarIsInitializedOptotal*
_output_shapes
: 
M
total/AssignAssignVariableOptotaltotal/Initializer/zeros*
dtype0
W
total/Read/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
v
count/Initializer/zerosConst*
valueB
 *    *
_class

loc:@count*
dtype0*
_output_shapes
: 
x
countVarHandleOp*
shape: *
shared_namecount*
_class

loc:@count*
dtype0*
_output_shapes
: 
[
&count/IsInitialized/VarIsInitializedOpVarIsInitializedOpcount*
_output_shapes
: 
M
count/AssignAssignVariableOpcountcount/Initializer/zeros*
dtype0
W
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_output_shapes
: 
|
metrics/acc/SqueezeSqueezedense_1_target*#
_output_shapes
:€€€€€€€€€*
squeeze_dims

€€€€€€€€€*
T0
g
metrics/acc/ArgMax/dimensionConst*
dtype0*
_output_shapes
: *
valueB :
€€€€€€€€€
y
metrics/acc/ArgMaxArgMaxdense_1/Softmaxmetrics/acc/ArgMax/dimension*#
_output_shapes
:€€€€€€€€€*
T0
i
metrics/acc/CastCastmetrics/acc/ArgMax*

SrcT0	*#
_output_shapes
:€€€€€€€€€*

DstT0
o
metrics/acc/EqualEqualmetrics/acc/Squeezemetrics/acc/Cast*
T0*#
_output_shapes
:€€€€€€€€€
j
metrics/acc/Cast_1Castmetrics/acc/Equal*#
_output_shapes
:€€€€€€€€€*

DstT0*

SrcT0

[
metrics/acc/ConstConst*
valueB: *
dtype0*
_output_shapes
:
^
metrics/acc/SumSummetrics/acc/Cast_1metrics/acc/Const*
T0*
_output_shapes
: 
[
metrics/acc/AssignAddVariableOpAssignAddVariableOptotalmetrics/acc/Sum*
dtype0
М
metrics/acc/ReadVariableOpReadVariableOptotal ^metrics/acc/AssignAddVariableOp^metrics/acc/Sum*
dtype0*
_output_shapes
: 
M
metrics/acc/SizeSizemetrics/acc/Cast_1*
T0*
_output_shapes
: 
\
metrics/acc/Cast_2Castmetrics/acc/Size*

SrcT0*
_output_shapes
: *

DstT0
В
!metrics/acc/AssignAddVariableOp_1AssignAddVariableOpcountmetrics/acc/Cast_2 ^metrics/acc/AssignAddVariableOp*
dtype0
†
metrics/acc/ReadVariableOp_1ReadVariableOpcount ^metrics/acc/AssignAddVariableOp"^metrics/acc/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
З
%metrics/acc/div_no_nan/ReadVariableOpReadVariableOptotal"^metrics/acc/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
Й
'metrics/acc/div_no_nan/ReadVariableOp_1ReadVariableOpcount"^metrics/acc/AssignAddVariableOp_1*
dtype0*
_output_shapes
: 
У
metrics/acc/div_no_nanDivNoNan%metrics/acc/div_no_nan/ReadVariableOp'metrics/acc/div_no_nan/ReadVariableOp_1*
T0*
_output_shapes
: 
Y
metrics/acc/IdentityIdentitymetrics/acc/div_no_nan*
T0*
_output_shapes
: 
x
loss/dense_1_loss/CastCastdense_1_target*

SrcT0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*

DstT0	
V
loss/dense_1_loss/ShapeShapedense_1/BiasAdd*
T0*
_output_shapes
:
r
loss/dense_1_loss/Reshape/shapeConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
Л
loss/dense_1_loss/ReshapeReshapeloss/dense_1_loss/Castloss/dense_1_loss/Reshape/shape*
T0	*#
_output_shapes
:€€€€€€€€€
x
%loss/dense_1_loss/strided_slice/stackConst*
valueB:
€€€€€€€€€*
dtype0*
_output_shapes
:
q
'loss/dense_1_loss/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
q
'loss/dense_1_loss/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
З
loss/dense_1_loss/strided_sliceStridedSliceloss/dense_1_loss/Shape%loss/dense_1_loss/strided_slice/stack'loss/dense_1_loss/strided_slice/stack_1'loss/dense_1_loss/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
n
#loss/dense_1_loss/Reshape_1/shape/0Const*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
Э
!loss/dense_1_loss/Reshape_1/shapePack#loss/dense_1_loss/Reshape_1/shape/0loss/dense_1_loss/strided_slice*
T0*
N*
_output_shapes
:
Х
loss/dense_1_loss/Reshape_1Reshapedense_1/BiasAdd!loss/dense_1_loss/Reshape_1/shape*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
Д
;loss/dense_1_loss/SparseSoftmaxCrossEntropyWithLogits/ShapeShapeloss/dense_1_loss/Reshape*
T0	*
_output_shapes
:
В
Yloss/dense_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitsloss/dense_1_loss/Reshape_1loss/dense_1_loss/Reshape*
T0*?
_output_shapes-
+:€€€€€€€€€:€€€€€€€€€€€€€€€€€€
k
&loss/dense_1_loss/weighted_loss/Cast/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?
Ч
Tloss/dense_1_loss/weighted_loss/broadcast_weights/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
Х
Sloss/dense_1_loss/weighted_loss/broadcast_weights/assert_broadcastable/weights/rankConst*
dtype0*
_output_shapes
: *
value	B : 
№
Sloss/dense_1_loss/weighted_loss/broadcast_weights/assert_broadcastable/values/shapeShapeYloss/dense_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits*
T0*
_output_shapes
:
Ф
Rloss/dense_1_loss/weighted_loss/broadcast_weights/assert_broadcastable/values/rankConst*
dtype0*
_output_shapes
: *
value	B :
j
bloss/dense_1_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOp
ѓ
Aloss/dense_1_loss/weighted_loss/broadcast_weights/ones_like/ShapeShapeYloss/dense_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitsc^loss/dense_1_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
_output_shapes
:*
T0
л
Aloss/dense_1_loss/weighted_loss/broadcast_weights/ones_like/ConstConstc^loss/dense_1_loss/weighted_loss/broadcast_weights/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *  А?
ч
;loss/dense_1_loss/weighted_loss/broadcast_weights/ones_likeFillAloss/dense_1_loss/weighted_loss/broadcast_weights/ones_like/ShapeAloss/dense_1_loss/weighted_loss/broadcast_weights/ones_like/Const*
T0*#
_output_shapes
:€€€€€€€€€
Ћ
1loss/dense_1_loss/weighted_loss/broadcast_weightsMul&loss/dense_1_loss/weighted_loss/Cast/x;loss/dense_1_loss/weighted_loss/broadcast_weights/ones_like*
T0*#
_output_shapes
:€€€€€€€€€
ж
#loss/dense_1_loss/weighted_loss/MulMulYloss/dense_1_loss/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits1loss/dense_1_loss/weighted_loss/broadcast_weights*
T0*#
_output_shapes
:€€€€€€€€€
a
loss/dense_1_loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:
{
loss/dense_1_loss/SumSum#loss/dense_1_loss/weighted_loss/Mulloss/dense_1_loss/Const*
_output_shapes
: *
T0
l
loss/dense_1_loss/num_elementsSize#loss/dense_1_loss/weighted_loss/Mul*
T0*
_output_shapes
: 
{
#loss/dense_1_loss/num_elements/CastCastloss/dense_1_loss/num_elements*
_output_shapes
: *

DstT0*

SrcT0
\
loss/dense_1_loss/Const_1Const*
valueB *
dtype0*
_output_shapes
: 
q
loss/dense_1_loss/Sum_1Sumloss/dense_1_loss/Sumloss/dense_1_loss/Const_1*
T0*
_output_shapes
: 
В
loss/dense_1_loss/valueDivNoNanloss/dense_1_loss/Sum_1#loss/dense_1_loss/num_elements/Cast*
_output_shapes
: *
T0
O

loss/mul/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
U
loss/mulMul
loss/mul/xloss/dense_1_loss/value*
_output_shapes
: *
T0
(
evaluation/group_depsNoOp	^loss/mul
Z
ConstConst"/device:CPU:0*
dtype0*
_output_shapes
: *
valueB Bmodel
э
RestoreV2/tensor_namesConst"/device:CPU:0*£
valueЩBЦB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/global_step/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
z
RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B B B B *
dtype0*
_output_shapes
:
Я
	RestoreV2	RestoreV2ConstRestoreV2/tensor_namesRestoreV2/shape_and_slices"/device:CPU:0*(
_output_shapes
:::::*
dtypes	
2	
B
IdentityIdentity	RestoreV2*
_output_shapes
:*
T0
G
AssignVariableOpAssignVariableOp
dense/biasIdentity*
dtype0
F

Identity_1IdentityRestoreV2:1*
T0*
_output_shapes
:
M
AssignVariableOp_1AssignVariableOpdense/kernel
Identity_1*
dtype0
F

Identity_2IdentityRestoreV2:2*
T0*
_output_shapes
:
M
AssignVariableOp_2AssignVariableOpdense_1/bias
Identity_2*
dtype0
F

Identity_3IdentityRestoreV2:3*
T0*
_output_shapes
:
O
AssignVariableOp_3AssignVariableOpdense_1/kernel
Identity_3*
dtype0
F

Identity_4IdentityRestoreV2:4*
T0	*
_output_shapes
:
W
AssignVariableOp_4AssignVariableOpTFOptimizer/iterations
Identity_4*
dtype0	
X
VarIsInitializedOpVarIsInitializedOpTFOptimizer/iterations*
_output_shapes
: 
N
VarIsInitializedOp_1VarIsInitializedOp
dense/bias*
_output_shapes
: 
P
VarIsInitializedOp_2VarIsInitializedOpdense_1/bias*
_output_shapes
: 
I
VarIsInitializedOp_3VarIsInitializedOptotal*
_output_shapes
: 
R
VarIsInitializedOp_4VarIsInitializedOpdense_1/kernel*
_output_shapes
: 
I
VarIsInitializedOp_5VarIsInitializedOpcount*
_output_shapes
: 
P
VarIsInitializedOp_6VarIsInitializedOpdense/kernel*
_output_shapes
: 
Ґ
initNoOp^TFOptimizer/iterations/Assign^count/Assign^dense/bias/Assign^dense/kernel/Assign^dense_1/bias/Assign^dense_1/kernel/Assign^total/Assign
W
div_no_nan/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
Y
div_no_nan/ReadVariableOp_1ReadVariableOpcount*
dtype0*
_output_shapes
: 
o

div_no_nanDivNoNandiv_no_nan/ReadVariableOpdiv_no_nan/ReadVariableOp_1*
_output_shapes
: *
T0
C

Identity_5Identity
div_no_nan*
T0*
_output_shapes
: 
x
metric_op_wrapperConst"^metrics/acc/AssignAddVariableOp_1*
valueB *
dtype0*
_output_shapes
: 
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
dtype0*
_output_shapes
: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 
р
save/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:*£
valueЩBЦB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/global_step/.ATTRIBUTES/VARIABLE_VALUE
m
save/SaveV2/shape_and_slicesConst*
valueBB B B B B *
dtype0*
_output_shapes
:
Ґ
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesdense/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp*TFOptimizer/iterations/Read/ReadVariableOp*
dtypes	
2	
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
В
save/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*£
valueЩBЦB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/global_step/.ATTRIBUTES/VARIABLE_VALUE

save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B B B B *
dtype0*
_output_shapes
:
≥
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*(
_output_shapes
:::::*
dtypes	
2	
L
save/IdentityIdentitysave/RestoreV2*
_output_shapes
:*
T0
Q
save/AssignVariableOpAssignVariableOp
dense/biassave/Identity*
dtype0
P
save/Identity_1Identitysave/RestoreV2:1*
T0*
_output_shapes
:
W
save/AssignVariableOp_1AssignVariableOpdense/kernelsave/Identity_1*
dtype0
P
save/Identity_2Identitysave/RestoreV2:2*
T0*
_output_shapes
:
W
save/AssignVariableOp_2AssignVariableOpdense_1/biassave/Identity_2*
dtype0
P
save/Identity_3Identitysave/RestoreV2:3*
T0*
_output_shapes
:
Y
save/AssignVariableOp_3AssignVariableOpdense_1/kernelsave/Identity_3*
dtype0
P
save/Identity_4Identitysave/RestoreV2:4*
T0	*
_output_shapes
:
a
save/AssignVariableOp_4AssignVariableOpTFOptimizer/iterationssave/Identity_4*
dtype0	
Ш
save/restore_allNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_2^save/AssignVariableOp_3^save/AssignVariableOp_4
,
init_1NoOp^count/Assign^total/Assign"ЖD
save/Const:0save/control_dependency:0save/restore_all 5 @F8"µ
global_step•Ґ
Я
TFOptimizer/iterations:0TFOptimizer/iterations/Assign,TFOptimizer/iterations/Read/ReadVariableOp:0(22TFOptimizer/iterations/Initializer/initial_value:08"К
	variablesьщ
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08
А
dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2+dense_1/kernel/Initializer/random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08
Я
TFOptimizer/iterations:0TFOptimizer/iterations/Assign,TFOptimizer/iterations/Read/ReadVariableOp:0(22TFOptimizer/iterations/Initializer/initial_value:08"Ф
trainable_variablesьщ
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08
А
dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2+dense_1/kernel/Initializer/random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08
Я
TFOptimizer/iterations:0TFOptimizer/iterations/Assign,TFOptimizer/iterations/Read/ReadVariableOp:0(22TFOptimizer/iterations/Initializer/initial_value:08"≈
local_variables±Ѓ
U
total:0total/Assigntotal/Read/ReadVariableOp:0(2total/Initializer/zeros:0@H
U
count:0count/Assigncount/Read/ReadVariableOp:0(2count/Initializer/zeros:0@H*ў
eval–
4
dense_input%
dense_input:0€€€€€€€€€А
B
dense_1_target0
dense_1_target:0€€€€€€€€€€€€€€€€€€?
predictions/dense_1(
dense_1/Softmax:0€€€€€€€€€4
metrics/acc/update_op
metric_op_wrapper:0 '
metrics/acc/value
Identity_5:0 
loss

loss/mul:0 tensorflow/supervised/eval*@
__saved_model_init_op'%
__saved_model_init_op
init_1†g
э–
:
Add
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
9
Softmax
logits"T
softmax"T"
Ttype:
2
:
Sub
x"T
y"T
z"T"
Ttype:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ
9
VarIsInitializedOp
resource
is_initialized
И"serve*1.15.02v1.15.0-rc3-22-g590d6ee8ЧQ
p
dense_inputPlaceholder*
dtype0*(
_output_shapes
:€€€€€€€€€А*
shape:€€€€€€€€€А
Я
-dense/kernel/Initializer/random_uniform/shapeConst*
valueB"   @   *
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:
С
+dense/kernel/Initializer/random_uniform/minConst*
valueB
 *тк≠љ*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
С
+dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *тк≠=*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 
Ќ
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
T0*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:	А@
ќ
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*
_class
loc:@dense/kernel
б
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	А@
”
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	А@
Ц
dense/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:	А@*
shared_namedense/kernel*
_class
loc:@dense/kernel
i
-dense/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense/kernel*
_output_shapes
: 
k
dense/kernel/AssignAssignVariableOpdense/kernel'dense/kernel/Initializer/random_uniform*
dtype0
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes
:	А@
И
dense/bias/Initializer/zerosConst*
valueB@*    *
_class
loc:@dense/bias*
dtype0*
_output_shapes
:@
Л

dense/biasVarHandleOp*
shape:@*
shared_name
dense/bias*
_class
loc:@dense/bias*
dtype0*
_output_shapes
: 
e
+dense/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp
dense/bias*
_output_shapes
: 
\
dense/bias/AssignAssignVariableOp
dense/biasdense/bias/Initializer/zeros*
dtype0
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:@
i
dense/MatMul/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes
:	А@
r
dense/MatMulMatMuldense_inputdense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€@
c
dense/BiasAdd/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:@
v
dense/BiasAddBiasAdddense/MatMuldense/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€@
S

dense/ReluReludense/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€@
£
/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"@      *!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
:
Х
-dense_1/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *2uЖЊ*!
_class
loc:@dense_1/kernel
Х
-dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *2uЖ>*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: 
“
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*
T0*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes

:@
÷
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*!
_class
loc:@dense_1/kernel
и
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:@
Џ
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:@
Ы
dense_1/kernelVarHandleOp*!
_class
loc:@dense_1/kernel*
dtype0*
_output_shapes
: *
shape
:@*
shared_namedense_1/kernel
m
/dense_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/kernel*
_output_shapes
: 
q
dense_1/kernel/AssignAssignVariableOpdense_1/kernel)dense_1/kernel/Initializer/random_uniform*
dtype0
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes

:@
М
dense_1/bias/Initializer/zerosConst*
valueB*    *
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
:
С
dense_1/biasVarHandleOp*
shape:*
shared_namedense_1/bias*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: 
i
-dense_1/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense_1/bias*
_output_shapes
: 
b
dense_1/bias/AssignAssignVariableOpdense_1/biasdense_1/bias/Initializer/zeros*
dtype0
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes
:
l
dense_1/MatMul/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes

:@
u
dense_1/MatMulMatMul
dense/Reludense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€
g
dense_1/BiasAdd/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes
:
|
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/BiasAdd/ReadVariableOp*'
_output_shapes
:€€€€€€€€€*
T0
]
dense_1/SoftmaxSoftmaxdense_1/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
,
predict/group_depsNoOp^dense_1/Softmax
Z
ConstConst"/device:CPU:0*
valueB Bmodel*
dtype0*
_output_shapes
: 
Ћ
RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*с
valueзBдB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
x
RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueBB B B B *
dtype0*
_output_shapes
:
Ъ
	RestoreV2	RestoreV2ConstRestoreV2/tensor_namesRestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2*$
_output_shapes
::::
B
IdentityIdentity	RestoreV2*
T0*
_output_shapes
:
G
AssignVariableOpAssignVariableOp
dense/biasIdentity*
dtype0
F

Identity_1IdentityRestoreV2:1*
T0*
_output_shapes
:
M
AssignVariableOp_1AssignVariableOpdense/kernel
Identity_1*
dtype0
F

Identity_2IdentityRestoreV2:2*
T0*
_output_shapes
:
M
AssignVariableOp_2AssignVariableOpdense_1/bias
Identity_2*
dtype0
F

Identity_3IdentityRestoreV2:3*
T0*
_output_shapes
:
O
AssignVariableOp_3AssignVariableOpdense_1/kernel
Identity_3*
dtype0
N
VarIsInitializedOpVarIsInitializedOpdense/kernel*
_output_shapes
: 
R
VarIsInitializedOp_1VarIsInitializedOpdense_1/kernel*
_output_shapes
: 
N
VarIsInitializedOp_2VarIsInitializedOp
dense/bias*
_output_shapes
: 
P
VarIsInitializedOp_3VarIsInitializedOpdense_1/bias*
_output_shapes
: 
d
initNoOp^dense/bias/Assign^dense/kernel/Assign^dense_1/bias/Assign^dense_1/kernel/Assign
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
_output_shapes
: *
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 
Њ
save/SaveV2/tensor_namesConst*с
valueзBдB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
k
save/SaveV2/shape_and_slicesConst*
valueBB B B B *
dtype0*
_output_shapes
:
х
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesdense/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp*
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
–
save/RestoreV2/tensor_namesConst"/device:CPU:0*с
valueзBдB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
}
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueBB B B B 
Ѓ
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*$
_output_shapes
::::*
dtypes
2
L
save/IdentityIdentitysave/RestoreV2*
T0*
_output_shapes
:
Q
save/AssignVariableOpAssignVariableOp
dense/biassave/Identity*
dtype0
P
save/Identity_1Identitysave/RestoreV2:1*
T0*
_output_shapes
:
W
save/AssignVariableOp_1AssignVariableOpdense/kernelsave/Identity_1*
dtype0
P
save/Identity_2Identitysave/RestoreV2:2*
T0*
_output_shapes
:
W
save/AssignVariableOp_2AssignVariableOpdense_1/biassave/Identity_2*
dtype0
P
save/Identity_3Identitysave/RestoreV2:3*
T0*
_output_shapes
:
Y
save/AssignVariableOp_3AssignVariableOpdense_1/kernelsave/Identity_3*
dtype0
~
save/restore_allNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_2^save/AssignVariableOp_3

init_1NoOp"ЖD
save/Const:0save/control_dependency:0save/restore_all 5 @F8"т
trainable_variablesЏ„
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08
А
dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2+dense_1/kernel/Initializer/random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08"и
	variablesЏ„
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08
А
dense_1/kernel:0dense_1/kernel/Assign$dense_1/kernel/Read/ReadVariableOp:0(2+dense_1/kernel/Initializer/random_uniform:08
o
dense_1/bias:0dense_1/bias/Assign"dense_1/bias/Read/ReadVariableOp:0(2 dense_1/bias/Initializer/zeros:08*Ы
serving_defaultЗ
4
dense_input%
dense_input:0€€€€€€€€€А3
dense_1(
dense_1/Softmax:0€€€€€€€€€tensorflow/serving/predict*@
__saved_model_init_op'%
__saved_model_init_op
init_1