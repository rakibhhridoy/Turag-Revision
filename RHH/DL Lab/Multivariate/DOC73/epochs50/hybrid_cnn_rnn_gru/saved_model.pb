8
"ъ!
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
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

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

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

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
Г
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
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
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
С
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ј

StatelessWhile

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint

@
StaticRegexFullMatch	
input

output
"
patternstring
ї
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

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
<
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
А
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleщшelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleщшelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintџџџџџџџџџ
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8іЛ5

Adam/gru_9/gru_cell_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*-
shared_nameAdam/gru_9/gru_cell_9/bias/v

0Adam/gru_9/gru_cell_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/gru_9/gru_cell_9/bias/v*
_output_shapes

:`*
dtype0
Ќ
(Adam/gru_9/gru_cell_9/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: `*9
shared_name*(Adam/gru_9/gru_cell_9/recurrent_kernel/v
Ѕ
<Adam/gru_9/gru_cell_9/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp(Adam/gru_9/gru_cell_9/recurrent_kernel/v*
_output_shapes

: `*
dtype0

Adam/gru_9/gru_cell_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@`*/
shared_name Adam/gru_9/gru_cell_9/kernel/v

2Adam/gru_9/gru_cell_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gru_9/gru_cell_9/kernel/v*
_output_shapes

:@`*
dtype0
Ќ
*Adam/simple_rnn_6/simple_rnn_cell_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/simple_rnn_6/simple_rnn_cell_6/bias/v
Ѕ
>Adam/simple_rnn_6/simple_rnn_cell_6/bias/v/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_6/simple_rnn_cell_6/bias/v*
_output_shapes
:@*
dtype0
Ш
6Adam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*G
shared_name86Adam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/v
С
JAdam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/v*
_output_shapes

:@@*
dtype0
Е
,Adam/simple_rnn_6/simple_rnn_cell_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	р@*=
shared_name.,Adam/simple_rnn_6/simple_rnn_cell_6/kernel/v
Ў
@Adam/simple_rnn_6/simple_rnn_cell_6/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_6/simple_rnn_cell_6/kernel/v*
_output_shapes
:	р@*
dtype0

Adam/dense_29/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_29/bias/v
y
(Adam/dense_29/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_29/bias/v*
_output_shapes
:*
dtype0

Adam/dense_29/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_29/kernel/v

*Adam/dense_29/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_29/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_28/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_28/bias/v
y
(Adam/dense_28/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_28/bias/v*
_output_shapes
:*
dtype0

Adam/dense_28/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_28/kernel/v

*Adam/dense_28/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_28/kernel/v*
_output_shapes

: *
dtype0

Adam/conv1d_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv1d_8/bias/v
y
(Adam/conv1d_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_8/bias/v*
_output_shapes
: *
dtype0

Adam/conv1d_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv1d_8/kernel/v

*Adam/conv1d_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_8/kernel/v*"
_output_shapes
: *
dtype0

Adam/gru_9/gru_cell_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*-
shared_nameAdam/gru_9/gru_cell_9/bias/m

0Adam/gru_9/gru_cell_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/gru_9/gru_cell_9/bias/m*
_output_shapes

:`*
dtype0
Ќ
(Adam/gru_9/gru_cell_9/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: `*9
shared_name*(Adam/gru_9/gru_cell_9/recurrent_kernel/m
Ѕ
<Adam/gru_9/gru_cell_9/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp(Adam/gru_9/gru_cell_9/recurrent_kernel/m*
_output_shapes

: `*
dtype0

Adam/gru_9/gru_cell_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@`*/
shared_name Adam/gru_9/gru_cell_9/kernel/m

2Adam/gru_9/gru_cell_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gru_9/gru_cell_9/kernel/m*
_output_shapes

:@`*
dtype0
Ќ
*Adam/simple_rnn_6/simple_rnn_cell_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/simple_rnn_6/simple_rnn_cell_6/bias/m
Ѕ
>Adam/simple_rnn_6/simple_rnn_cell_6/bias/m/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_6/simple_rnn_cell_6/bias/m*
_output_shapes
:@*
dtype0
Ш
6Adam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*G
shared_name86Adam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/m
С
JAdam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/m*
_output_shapes

:@@*
dtype0
Е
,Adam/simple_rnn_6/simple_rnn_cell_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	р@*=
shared_name.,Adam/simple_rnn_6/simple_rnn_cell_6/kernel/m
Ў
@Adam/simple_rnn_6/simple_rnn_cell_6/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_6/simple_rnn_cell_6/kernel/m*
_output_shapes
:	р@*
dtype0

Adam/dense_29/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_29/bias/m
y
(Adam/dense_29/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_29/bias/m*
_output_shapes
:*
dtype0

Adam/dense_29/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_29/kernel/m

*Adam/dense_29/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_29/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_28/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_28/bias/m
y
(Adam/dense_28/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_28/bias/m*
_output_shapes
:*
dtype0

Adam/dense_28/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_28/kernel/m

*Adam/dense_28/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_28/kernel/m*
_output_shapes

: *
dtype0

Adam/conv1d_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv1d_8/bias/m
y
(Adam/conv1d_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_8/bias/m*
_output_shapes
: *
dtype0

Adam/conv1d_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv1d_8/kernel/m

*Adam/conv1d_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_8/kernel/m*"
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	

gru_9/gru_cell_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*&
shared_namegru_9/gru_cell_9/bias

)gru_9/gru_cell_9/bias/Read/ReadVariableOpReadVariableOpgru_9/gru_cell_9/bias*
_output_shapes

:`*
dtype0

!gru_9/gru_cell_9/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: `*2
shared_name#!gru_9/gru_cell_9/recurrent_kernel

5gru_9/gru_cell_9/recurrent_kernel/Read/ReadVariableOpReadVariableOp!gru_9/gru_cell_9/recurrent_kernel*
_output_shapes

: `*
dtype0

gru_9/gru_cell_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@`*(
shared_namegru_9/gru_cell_9/kernel

+gru_9/gru_cell_9/kernel/Read/ReadVariableOpReadVariableOpgru_9/gru_cell_9/kernel*
_output_shapes

:@`*
dtype0

#simple_rnn_6/simple_rnn_cell_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#simple_rnn_6/simple_rnn_cell_6/bias

7simple_rnn_6/simple_rnn_cell_6/bias/Read/ReadVariableOpReadVariableOp#simple_rnn_6/simple_rnn_cell_6/bias*
_output_shapes
:@*
dtype0
К
/simple_rnn_6/simple_rnn_cell_6/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*@
shared_name1/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel
Г
Csimple_rnn_6/simple_rnn_cell_6/recurrent_kernel/Read/ReadVariableOpReadVariableOp/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel*
_output_shapes

:@@*
dtype0
Ї
%simple_rnn_6/simple_rnn_cell_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	р@*6
shared_name'%simple_rnn_6/simple_rnn_cell_6/kernel
 
9simple_rnn_6/simple_rnn_cell_6/kernel/Read/ReadVariableOpReadVariableOp%simple_rnn_6/simple_rnn_cell_6/kernel*
_output_shapes
:	р@*
dtype0
r
dense_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_29/bias
k
!dense_29/bias/Read/ReadVariableOpReadVariableOpdense_29/bias*
_output_shapes
:*
dtype0
z
dense_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_29/kernel
s
#dense_29/kernel/Read/ReadVariableOpReadVariableOpdense_29/kernel*
_output_shapes

:*
dtype0
r
dense_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_28/bias
k
!dense_28/bias/Read/ReadVariableOpReadVariableOpdense_28/bias*
_output_shapes
:*
dtype0
z
dense_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_28/kernel
s
#dense_28/kernel/Read/ReadVariableOpReadVariableOpdense_28/kernel*
_output_shapes

: *
dtype0
r
conv1d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1d_8/bias
k
!conv1d_8/bias/Read/ReadVariableOpReadVariableOpconv1d_8/bias*
_output_shapes
: *
dtype0
~
conv1d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv1d_8/kernel
w
#conv1d_8/kernel/Read/ReadVariableOpReadVariableOpconv1d_8/kernel*"
_output_shapes
: *
dtype0

serving_default_conv1d_8_inputPlaceholder*+
_output_shapes
:џџџџџџџџџ*
dtype0* 
shape:џџџџџџџџџ

StatefulPartitionedCallStatefulPartitionedCallserving_default_conv1d_8_inputconv1d_8/kernelconv1d_8/bias%simple_rnn_6/simple_rnn_cell_6/kernel#simple_rnn_6/simple_rnn_cell_6/bias/simple_rnn_6/simple_rnn_cell_6/recurrent_kernelgru_9/gru_cell_9/kernel!gru_9/gru_cell_9/recurrent_kernelgru_9/gru_cell_9/biasdense_28/kerneldense_28/biasdense_29/kerneldense_29/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_1930846

NoOpNoOp
Њ`
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*х_
valueл_Bи_ Bб_
Љ
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
Ш
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 

 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses* 
Њ
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses
,cell
-
state_spec*
С
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses
4_random_generator
5cell
6
state_spec*
І
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses

=kernel
>bias*
І
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses

Ekernel
Fbias*
Z
0
1
G2
H3
I4
J5
K6
L7
=8
>9
E10
F11*
Z
0
1
G2
H3
I4
J5
K6
L7
=8
>9
E10
F11*
* 
А
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Rtrace_0
Strace_1
Ttrace_2
Utrace_3* 
6
Vtrace_0
Wtrace_1
Xtrace_2
Ytrace_3* 
* 
Д
Ziter

[beta_1

\beta_2
	]decay
^learning_ratemХmЦ=mЧ>mШEmЩFmЪGmЫHmЬImЭJmЮKmЯLmаvбvв=vг>vдEvеFvжGvзHvиIvйJvкKvлLvм*

_serving_default* 

0
1*

0
1*
* 

`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

etrace_0* 

ftrace_0* 
_Y
VARIABLE_VALUEconv1d_8/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv1d_8/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

ltrace_0* 

mtrace_0* 
* 
* 
* 

nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses* 

strace_0* 

ttrace_0* 

G0
H1
I2*

G0
H1
I2*
* 


ustates
vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*
6
{trace_0
|trace_1
}trace_2
~trace_3* 
9
trace_0
trace_1
trace_2
trace_3* 
к
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator

Gkernel
Hrecurrent_kernel
Ibias*
* 

J0
K1
L2*

J0
K1
L2*
* 
Ѕ
states
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses*
:
trace_0
trace_1
trace_2
trace_3* 
:
trace_0
trace_1
trace_2
trace_3* 
* 
к
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator

Jkernel
Krecurrent_kernel
Lbias*
* 

=0
>1*

=0
>1*
* 

non_trainable_variables
 layers
Ёmetrics
 Ђlayer_regularization_losses
Ѓlayer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses*

Єtrace_0* 

Ѕtrace_0* 
_Y
VARIABLE_VALUEdense_28/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_28/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

E0
F1*

E0
F1*
* 

Іnon_trainable_variables
Їlayers
Јmetrics
 Љlayer_regularization_losses
Њlayer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses*

Ћtrace_0* 

Ќtrace_0* 
_Y
VARIABLE_VALUEdense_29/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_29/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%simple_rnn_6/simple_rnn_cell_6/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#simple_rnn_6/simple_rnn_cell_6/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEgru_9/gru_cell_9/kernel&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!gru_9/gru_cell_9/recurrent_kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEgru_9/gru_cell_9/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
* 
5
0
1
2
3
4
5
6*

­0
Ў1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

,0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

G0
H1
I2*

G0
H1
I2*
* 

Џnon_trainable_variables
Аlayers
Бmetrics
 Вlayer_regularization_losses
Гlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

Дtrace_0
Еtrace_1* 

Жtrace_0
Зtrace_1* 
* 
* 
* 

50*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

J0
K1
L2*

J0
K1
L2*
* 

Иnon_trainable_variables
Йlayers
Кmetrics
 Лlayer_regularization_losses
Мlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
Н	variables
О	keras_api

Пtotal

Рcount*
<
С	variables
Т	keras_api

Уtotal

Фcount*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

П0
Р1*

Н	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

У0
Ф1*

С	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/conv1d_8/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv1d_8/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_28/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_28/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_29/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_29/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/simple_rnn_6/simple_rnn_cell_6/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE6Adam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/simple_rnn_6/simple_rnn_cell_6/bias/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/gru_9/gru_cell_9/kernel/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE(Adam/gru_9/gru_cell_9/recurrent_kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/gru_9/gru_cell_9/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/conv1d_8/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv1d_8/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_28/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_28/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_29/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_29/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/simple_rnn_6/simple_rnn_cell_6/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE6Adam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/simple_rnn_6/simple_rnn_cell_6/bias/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/gru_9/gru_cell_9/kernel/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE(Adam/gru_9/gru_cell_9/recurrent_kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/gru_9/gru_cell_9/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ъ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv1d_8/kernel/Read/ReadVariableOp!conv1d_8/bias/Read/ReadVariableOp#dense_28/kernel/Read/ReadVariableOp!dense_28/bias/Read/ReadVariableOp#dense_29/kernel/Read/ReadVariableOp!dense_29/bias/Read/ReadVariableOp9simple_rnn_6/simple_rnn_cell_6/kernel/Read/ReadVariableOpCsimple_rnn_6/simple_rnn_cell_6/recurrent_kernel/Read/ReadVariableOp7simple_rnn_6/simple_rnn_cell_6/bias/Read/ReadVariableOp+gru_9/gru_cell_9/kernel/Read/ReadVariableOp5gru_9/gru_cell_9/recurrent_kernel/Read/ReadVariableOp)gru_9/gru_cell_9/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/conv1d_8/kernel/m/Read/ReadVariableOp(Adam/conv1d_8/bias/m/Read/ReadVariableOp*Adam/dense_28/kernel/m/Read/ReadVariableOp(Adam/dense_28/bias/m/Read/ReadVariableOp*Adam/dense_29/kernel/m/Read/ReadVariableOp(Adam/dense_29/bias/m/Read/ReadVariableOp@Adam/simple_rnn_6/simple_rnn_cell_6/kernel/m/Read/ReadVariableOpJAdam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/m/Read/ReadVariableOp>Adam/simple_rnn_6/simple_rnn_cell_6/bias/m/Read/ReadVariableOp2Adam/gru_9/gru_cell_9/kernel/m/Read/ReadVariableOp<Adam/gru_9/gru_cell_9/recurrent_kernel/m/Read/ReadVariableOp0Adam/gru_9/gru_cell_9/bias/m/Read/ReadVariableOp*Adam/conv1d_8/kernel/v/Read/ReadVariableOp(Adam/conv1d_8/bias/v/Read/ReadVariableOp*Adam/dense_28/kernel/v/Read/ReadVariableOp(Adam/dense_28/bias/v/Read/ReadVariableOp*Adam/dense_29/kernel/v/Read/ReadVariableOp(Adam/dense_29/bias/v/Read/ReadVariableOp@Adam/simple_rnn_6/simple_rnn_cell_6/kernel/v/Read/ReadVariableOpJAdam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/v/Read/ReadVariableOp>Adam/simple_rnn_6/simple_rnn_cell_6/bias/v/Read/ReadVariableOp2Adam/gru_9/gru_cell_9/kernel/v/Read/ReadVariableOp<Adam/gru_9/gru_cell_9/recurrent_kernel/v/Read/ReadVariableOp0Adam/gru_9/gru_cell_9/bias/v/Read/ReadVariableOpConst*:
Tin3
12/	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__traced_save_1934267
с
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d_8/kernelconv1d_8/biasdense_28/kerneldense_28/biasdense_29/kerneldense_29/bias%simple_rnn_6/simple_rnn_cell_6/kernel/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel#simple_rnn_6/simple_rnn_cell_6/biasgru_9/gru_cell_9/kernel!gru_9/gru_cell_9/recurrent_kernelgru_9/gru_cell_9/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/conv1d_8/kernel/mAdam/conv1d_8/bias/mAdam/dense_28/kernel/mAdam/dense_28/bias/mAdam/dense_29/kernel/mAdam/dense_29/bias/m,Adam/simple_rnn_6/simple_rnn_cell_6/kernel/m6Adam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/m*Adam/simple_rnn_6/simple_rnn_cell_6/bias/mAdam/gru_9/gru_cell_9/kernel/m(Adam/gru_9/gru_cell_9/recurrent_kernel/mAdam/gru_9/gru_cell_9/bias/mAdam/conv1d_8/kernel/vAdam/conv1d_8/bias/vAdam/dense_28/kernel/vAdam/dense_28/bias/vAdam/dense_29/kernel/vAdam/dense_29/bias/v,Adam/simple_rnn_6/simple_rnn_cell_6/kernel/v6Adam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/v*Adam/simple_rnn_6/simple_rnn_cell_6/bias/vAdam/gru_9/gru_cell_9/kernel/v(Adam/gru_9/gru_cell_9/recurrent_kernel/vAdam/gru_9/gru_cell_9/bias/v*9
Tin2
02.*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__traced_restore_1934412Не3
ПЖ
ђ
#__inference__traced_restore_1934412
file_prefix6
 assignvariableop_conv1d_8_kernel: .
 assignvariableop_1_conv1d_8_bias: 4
"assignvariableop_2_dense_28_kernel: .
 assignvariableop_3_dense_28_bias:4
"assignvariableop_4_dense_29_kernel:.
 assignvariableop_5_dense_29_bias:K
8assignvariableop_6_simple_rnn_6_simple_rnn_cell_6_kernel:	р@T
Bassignvariableop_7_simple_rnn_6_simple_rnn_cell_6_recurrent_kernel:@@D
6assignvariableop_8_simple_rnn_6_simple_rnn_cell_6_bias:@<
*assignvariableop_9_gru_9_gru_cell_9_kernel:@`G
5assignvariableop_10_gru_9_gru_cell_9_recurrent_kernel: `;
)assignvariableop_11_gru_9_gru_cell_9_bias:`'
assignvariableop_12_adam_iter:	 )
assignvariableop_13_adam_beta_1: )
assignvariableop_14_adam_beta_2: (
assignvariableop_15_adam_decay: 0
&assignvariableop_16_adam_learning_rate: %
assignvariableop_17_total_1: %
assignvariableop_18_count_1: #
assignvariableop_19_total: #
assignvariableop_20_count: @
*assignvariableop_21_adam_conv1d_8_kernel_m: 6
(assignvariableop_22_adam_conv1d_8_bias_m: <
*assignvariableop_23_adam_dense_28_kernel_m: 6
(assignvariableop_24_adam_dense_28_bias_m:<
*assignvariableop_25_adam_dense_29_kernel_m:6
(assignvariableop_26_adam_dense_29_bias_m:S
@assignvariableop_27_adam_simple_rnn_6_simple_rnn_cell_6_kernel_m:	р@\
Jassignvariableop_28_adam_simple_rnn_6_simple_rnn_cell_6_recurrent_kernel_m:@@L
>assignvariableop_29_adam_simple_rnn_6_simple_rnn_cell_6_bias_m:@D
2assignvariableop_30_adam_gru_9_gru_cell_9_kernel_m:@`N
<assignvariableop_31_adam_gru_9_gru_cell_9_recurrent_kernel_m: `B
0assignvariableop_32_adam_gru_9_gru_cell_9_bias_m:`@
*assignvariableop_33_adam_conv1d_8_kernel_v: 6
(assignvariableop_34_adam_conv1d_8_bias_v: <
*assignvariableop_35_adam_dense_28_kernel_v: 6
(assignvariableop_36_adam_dense_28_bias_v:<
*assignvariableop_37_adam_dense_29_kernel_v:6
(assignvariableop_38_adam_dense_29_bias_v:S
@assignvariableop_39_adam_simple_rnn_6_simple_rnn_cell_6_kernel_v:	р@\
Jassignvariableop_40_adam_simple_rnn_6_simple_rnn_cell_6_recurrent_kernel_v:@@L
>assignvariableop_41_adam_simple_rnn_6_simple_rnn_cell_6_bias_v:@D
2assignvariableop_42_adam_gru_9_gru_cell_9_kernel_v:@`N
<assignvariableop_43_adam_gru_9_gru_cell_9_recurrent_kernel_v: `B
0assignvariableop_44_adam_gru_9_gru_cell_9_bias_v:`
identity_46ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*О
valueДBБ.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЬ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ю
_output_shapesЛ
И::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp assignvariableop_conv1d_8_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv1d_8_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_28_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_28_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_29_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_29_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_6AssignVariableOp8assignvariableop_6_simple_rnn_6_simple_rnn_cell_6_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_7AssignVariableOpBassignvariableop_7_simple_rnn_6_simple_rnn_cell_6_recurrent_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_8AssignVariableOp6assignvariableop_8_simple_rnn_6_simple_rnn_cell_6_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp*assignvariableop_9_gru_9_gru_cell_9_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOp_10AssignVariableOp5assignvariableop_10_gru_9_gru_cell_9_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp)assignvariableop_11_gru_9_gru_cell_9_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_conv1d_8_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_conv1d_8_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_28_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_28_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_29_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_29_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_27AssignVariableOp@assignvariableop_27_adam_simple_rnn_6_simple_rnn_cell_6_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_28AssignVariableOpJassignvariableop_28_adam_simple_rnn_6_simple_rnn_cell_6_recurrent_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_29AssignVariableOp>assignvariableop_29_adam_simple_rnn_6_simple_rnn_cell_6_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Ѓ
AssignVariableOp_30AssignVariableOp2assignvariableop_30_adam_gru_9_gru_cell_9_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_31AssignVariableOp<assignvariableop_31_adam_gru_9_gru_cell_9_recurrent_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Ё
AssignVariableOp_32AssignVariableOp0assignvariableop_32_adam_gru_9_gru_cell_9_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_conv1d_8_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_conv1d_8_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_28_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_28_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_29_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_29_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_39AssignVariableOp@assignvariableop_39_adam_simple_rnn_6_simple_rnn_cell_6_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_40AssignVariableOpJassignvariableop_40_adam_simple_rnn_6_simple_rnn_cell_6_recurrent_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_41AssignVariableOp>assignvariableop_41_adam_simple_rnn_6_simple_rnn_cell_6_bias_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:Ѓ
AssignVariableOp_42AssignVariableOp2assignvariableop_42_adam_gru_9_gru_cell_9_kernel_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_43AssignVariableOp<assignvariableop_43_adam_gru_9_gru_cell_9_recurrent_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:Ё
AssignVariableOp_44AssignVariableOp0assignvariableop_44_adam_gru_9_gru_cell_9_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ­
Identity_45Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_46IdentityIdentity_45:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_46Identity_46:output:0*o
_input_shapes^
\: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
е"
С
J__inference_sequential_21_layer_call_and_return_conditional_losses_1930024

inputs&
conv1d_8_1929477: 
conv1d_8_1929479: '
simple_rnn_6_1929599:	р@"
simple_rnn_6_1929601:@&
simple_rnn_6_1929603:@@
gru_9_1929984:@`
gru_9_1929986: `
gru_9_1929988:`"
dense_28_1930002: 
dense_28_1930004:"
dense_29_1930018:
dense_29_1930020:
identityЂ conv1d_8/StatefulPartitionedCallЂ dense_28/StatefulPartitionedCallЂ dense_29/StatefulPartitionedCallЂgru_9/StatefulPartitionedCallЂ$simple_rnn_6/StatefulPartitionedCallї
 conv1d_8/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_8_1929477conv1d_8_1929479*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv1d_8_layer_call_and_return_conditional_losses_1929476п
flatten_9/PartitionedCallPartitionedCall)conv1d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџр* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_9_layer_call_and_return_conditional_losses_1929488ш
repeat_vector_2/PartitionedCallPartitionedCall"flatten_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџр* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_repeat_vector_2_layer_call_and_return_conditional_losses_1928378С
$simple_rnn_6/StatefulPartitionedCallStatefulPartitionedCall(repeat_vector_2/PartitionedCall:output:0simple_rnn_6_1929599simple_rnn_6_1929601simple_rnn_6_1929603*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_1929598
gru_9/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_6/StatefulPartitionedCall:output:0gru_9_1929984gru_9_1929986gru_9_1929988*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_gru_9_layer_call_and_return_conditional_losses_1929983
 dense_28/StatefulPartitionedCallStatefulPartitionedCall&gru_9/StatefulPartitionedCall:output:0dense_28_1930002dense_28_1930004*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_28_layer_call_and_return_conditional_losses_1930001
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0dense_29_1930018dense_29_1930020*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_29_layer_call_and_return_conditional_losses_1930017x
IdentityIdentity)dense_29/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџі
NoOpNoOp!^conv1d_8/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall^gru_9/StatefulPartitionedCall%^simple_rnn_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ: : : : : : : : : : : : 2D
 conv1d_8/StatefulPartitionedCall conv1d_8/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2>
gru_9/StatefulPartitionedCallgru_9/StatefulPartitionedCall2L
$simple_rnn_6/StatefulPartitionedCall$simple_rnn_6/StatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

ь
N__inference_simple_rnn_cell_6_layer_call_and_return_conditional_losses_1934092

inputs
states_01
matmul_readvariableop_resource:	р@-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	р@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@G
TanhTanhadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:џџџџџџџџџр:џџџџџџџџџ@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџр
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ@
"
_user_specified_name
states/0
Ћ=
Н
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_1930592

inputsC
0simple_rnn_cell_6_matmul_readvariableop_resource:	р@?
1simple_rnn_cell_6_biasadd_readvariableop_resource:@D
2simple_rnn_cell_6_matmul_1_readvariableop_resource:@@
identityЂ(simple_rnn_cell_6/BiasAdd/ReadVariableOpЂ'simple_rnn_cell_6/MatMul/ReadVariableOpЂ)simple_rnn_cell_6/MatMul_1/ReadVariableOpЂwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџрD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџр  р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџр*
shrink_axis_mask
'simple_rnn_cell_6/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_6_matmul_readvariableop_resource*
_output_shapes
:	р@*
dtype0
simple_rnn_cell_6/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
(simple_rnn_cell_6/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ќ
simple_rnn_cell_6/BiasAddBiasAdd"simple_rnn_cell_6/MatMul:product:00simple_rnn_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
)simple_rnn_cell_6/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_6_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_6/MatMul_1MatMulzeros:output:01simple_rnn_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
simple_rnn_cell_6/addAddV2"simple_rnn_cell_6/BiasAdd:output:0$simple_rnn_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@k
simple_rnn_cell_6/TanhTanhsimple_rnn_cell_6/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : к
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_6_matmul_readvariableop_resource1simple_rnn_cell_6_biasadd_readvariableop_resource2simple_rnn_cell_6_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1930526*
condR
while_cond_1930525*8
output_shapes'
%: : : : :џџџџџџџџџ@: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   Т
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ@Я
NoOpNoOp)^simple_rnn_cell_6/BiasAdd/ReadVariableOp(^simple_rnn_cell_6/MatMul/ReadVariableOp*^simple_rnn_cell_6/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџр: : : 2T
(simple_rnn_cell_6/BiasAdd/ReadVariableOp(simple_rnn_cell_6/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_6/MatMul/ReadVariableOp'simple_rnn_cell_6/MatMul/ReadVariableOp2V
)simple_rnn_cell_6/MatMul_1/ReadVariableOp)simple_rnn_cell_6/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:џџџџџџџџџр
 
_user_specified_nameinputs
Џ,
Ы
while_body_1932387
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
8while_simple_rnn_cell_6_matmul_readvariableop_resource_0:	р@G
9while_simple_rnn_cell_6_biasadd_readvariableop_resource_0:@L
:while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
6while_simple_rnn_cell_6_matmul_readvariableop_resource:	р@E
7while_simple_rnn_cell_6_biasadd_readvariableop_resource:@J
8while_simple_rnn_cell_6_matmul_1_readvariableop_resource:@@Ђ.while/simple_rnn_cell_6/BiasAdd/ReadVariableOpЂ-while/simple_rnn_cell_6/MatMul/ReadVariableOpЂ/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџр  Ї
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџр*
element_dtype0Ї
-while/simple_rnn_cell_6/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_6_matmul_readvariableop_resource_0*
_output_shapes
:	р@*
dtype0У
while/simple_rnn_cell_6/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@Є
.while/simple_rnn_cell_6/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_6_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0О
while/simple_rnn_cell_6/BiasAddBiasAdd(while/simple_rnn_cell_6/MatMul:product:06while/simple_rnn_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@Њ
/while/simple_rnn_cell_6/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0Њ
 while/simple_rnn_cell_6/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@Ќ
while/simple_rnn_cell_6/addAddV2(while/simple_rnn_cell_6/BiasAdd:output:0*while/simple_rnn_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@w
while/simple_rnn_cell_6/TanhTanhwhile/simple_rnn_cell_6/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Щ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_6/Tanh:y:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: }
while/Identity_4Identity while/simple_rnn_cell_6/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@п

while/NoOpNoOp/^while/simple_rnn_cell_6/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_6/MatMul/ReadVariableOp0^while/simple_rnn_cell_6/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_6_biasadd_readvariableop_resource9while_simple_rnn_cell_6_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_6_matmul_1_readvariableop_resource:while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_6_matmul_readvariableop_resource8while_simple_rnn_cell_6_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ@: : : : : 2`
.while/simple_rnn_cell_6/BiasAdd/ReadVariableOp.while/simple_rnn_cell_6/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_6/MatMul/ReadVariableOp-while/simple_rnn_cell_6/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ@:

_output_shapes
: :

_output_shapes
: 
Щ4
А
)__inference_gpu_gru_with_fallback_1928215

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџV
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:РS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ъ
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:РIU
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    Ъ
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*H
_output_shapes6
4:џџџџџџџџџ :џџџџџџџџџ : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџ *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_1
ExpandDimsstrided_slice:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџ I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:џџџџџџџџџ@:џџџџџџџџџ :@`: `:`*<
api_implements*(gru_73e98d14-b186-4092-be4c-44eb47643dc0*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
Щ4
А
)__inference_gpu_gru_with_fallback_1931265

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџV
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:РS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ъ
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:РIU
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    Ъ
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*H
_output_shapes6
4:џџџџџџџџџ :џџџџџџџџџ : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџ *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_1
ExpandDimsstrided_slice:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџ I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:џџџџџџџџџ@:џџџџџџџџџ :@`: `:`*<
api_implements*(gru_7cdd8818-ca5e-4d7b-98a6-7015d31c0e59*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
п
Џ
while_cond_1932170
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1932170___redundant_placeholder05
1while_while_cond_1932170___redundant_placeholder15
1while_while_cond_1932170___redundant_placeholder25
1while_while_cond_1932170___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :џџџџџџџџџ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ@:

_output_shapes
: :

_output_shapes
:
н-
у
while_body_1929139
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ@*
element_dtype0
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ`{
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:џџџџџџџџџ`W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Е
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ`
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:џџџџџџџџџ`Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Л
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_splitr
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ Y
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ t
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ ]
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ o
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ k
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ U

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ l
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ P
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?k
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ c
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ h
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/add_3:z:0*
_output_shapes
: *
element_dtype0:щшвO
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ "4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?: : : : :џџџџџџџџџ : : :@`:`: `:`: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@`: 

_output_shapes
:`:$	 

_output_shapes

: `: 


_output_shapes
:`
э
Н
B__inference_gru_9_layer_call_and_return_conditional_losses_1929444

inputs.
read_readvariableop_resource:@`0
read_1_readvariableop_resource: `0
read_2_readvariableop_resource:`

identity_3ЂRead/ReadVariableOpЂRead_1/ReadVariableOpЂRead_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ p
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:@`*
dtype0Z
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:@`t
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

: `*
dtype0^

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

: `t
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes

:`*
dtype0^

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes

:`
PartitionedCallPartitionedCallinputszeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *S
_output_shapesA
?:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference_standard_gru_1929229i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ@: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
н-
у
while_body_1933704
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ@*
element_dtype0
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ`{
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:џџџџџџџџџ`W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Е
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ`
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:џџџџџџџџџ`Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Л
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_splitr
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ Y
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ t
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ ]
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ o
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ k
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ U

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ l
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ P
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?k
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ c
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ h
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/add_3:z:0*
_output_shapes
: *
element_dtype0:щшвO
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ "4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?: : : : :џџџџџџџџџ : : :@`:`: `:`: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@`: 

_output_shapes
:`:$	 

_output_shapes

: `: 


_output_shapes
:`

Й
.__inference_simple_rnn_6_layer_call_fn_1932010

inputs
unknown:	р@
	unknown_0:@
	unknown_1:@@
identityЂStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_1929598s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџр: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџр
 
_user_specified_nameinputs
л
Н
B__inference_gru_9_layer_call_and_return_conditional_losses_1933631

inputs.
read_readvariableop_resource:@`0
read_1_readvariableop_resource: `0
read_2_readvariableop_resource:`

identity_3ЂRead/ReadVariableOpЂRead_1/ReadVariableOpЂRead_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ p
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:@`*
dtype0Z
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:@`t
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

: `*
dtype0^

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

: `t
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes

:`*
dtype0^

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes

:`
PartitionedCallPartitionedCallinputszeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *S
_output_shapesA
?:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference_standard_gru_1933416i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ@: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ћ=
Н
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_1932345

inputsC
0simple_rnn_cell_6_matmul_readvariableop_resource:	р@?
1simple_rnn_cell_6_biasadd_readvariableop_resource:@D
2simple_rnn_cell_6_matmul_1_readvariableop_resource:@@
identityЂ(simple_rnn_cell_6/BiasAdd/ReadVariableOpЂ'simple_rnn_cell_6/MatMul/ReadVariableOpЂ)simple_rnn_cell_6/MatMul_1/ReadVariableOpЂwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџрD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџр  р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџр*
shrink_axis_mask
'simple_rnn_cell_6/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_6_matmul_readvariableop_resource*
_output_shapes
:	р@*
dtype0
simple_rnn_cell_6/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
(simple_rnn_cell_6/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ќ
simple_rnn_cell_6/BiasAddBiasAdd"simple_rnn_cell_6/MatMul:product:00simple_rnn_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
)simple_rnn_cell_6/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_6_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_6/MatMul_1MatMulzeros:output:01simple_rnn_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
simple_rnn_cell_6/addAddV2"simple_rnn_cell_6/BiasAdd:output:0$simple_rnn_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@k
simple_rnn_cell_6/TanhTanhsimple_rnn_cell_6/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : к
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_6_matmul_readvariableop_resource1simple_rnn_cell_6_biasadd_readvariableop_resource2simple_rnn_cell_6_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1932279*
condR
while_cond_1932278*8
output_shapes'
%: : : : :џџџџџџџџџ@: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   Т
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ@Я
NoOpNoOp)^simple_rnn_cell_6/BiasAdd/ReadVariableOp(^simple_rnn_cell_6/MatMul/ReadVariableOp*^simple_rnn_cell_6/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџр: : : 2T
(simple_rnn_cell_6/BiasAdd/ReadVariableOp(simple_rnn_cell_6/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_6/MatMul/ReadVariableOp'simple_rnn_cell_6/MatMul/ReadVariableOp2V
)simple_rnn_cell_6/MatMul_1/ReadVariableOp)simple_rnn_cell_6/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:џџџџџџџџџр
 
_user_specified_nameinputs
Ц
к

<__inference___backward_gpu_gru_with_fallback_1928917_1929053
placeholder
placeholder_1
placeholder_2
placeholder_33
/gradients_expanddims_1_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:џџџџџџџџџ d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:џџџџџџџџџ `
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:џџџџџџџџџ O
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 
!gradients/ExpandDims_1_grad/ShapeShape/gradients_expanddims_1_grad_shape_strided_slice*
T0*
_output_shapes
:Њ
#gradients/ExpandDims_1_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:Є
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Х
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_1_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:џџџџџџџџџ }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:Ђ
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *
shrink_axis_maska
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*T
_output_shapesB
@:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџ : :РI*
rnn_modegru
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:й
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:Х
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ \
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:g
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: 
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::ц
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:щ
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
: щ
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
: щ
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: щ
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: ь
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: ь
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   Ѓ
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   Ѕ
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   Ѕ
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ѕ
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ѕ
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ѕ
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  h
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ё
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ё
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ё
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ѓ
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Є
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Є
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: 
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:З
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ 
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:З
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ 
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:З
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ 
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:З
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:З
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:З
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Р
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:@`
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

: `m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   `   Ё
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:`{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ e

Identity_2Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes

:@`g

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes

: `h

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes

:`"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*Љ
_input_shapes
:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ : :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџџџџџџџџџџ : ::џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџ : :РI::џџџџџџџџџ : ::::::: : : *<
api_implements*(gru_e1cf211a-1d8c-47c6-ac5e-267262ea26df*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_gru_with_fallback_1929052*
go_backwards( *

time_major( :- )
'
_output_shapes
:џџџџџџџџџ :1-
+
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :1-
+
_output_shapes
:џџџџџџџџџ ::6
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ :

_output_shapes
: :

_output_shapes
:::	6
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@:1
-
+
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :!

_output_shapes	
:РI: 

_output_shapes
::-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
н-
у
while_body_1929678
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ@*
element_dtype0
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ`{
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:џџџџџџџџџ`W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Е
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ`
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:џџџџџџџџџ`Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Л
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_splitr
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ Y
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ t
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ ]
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ o
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ k
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ U

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ l
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ P
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?k
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ c
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ h
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/add_3:z:0*
_output_shapes
: *
element_dtype0:щшвO
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ "4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?: : : : :џџџџџџџџџ : : :@`:`: `:`: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@`: 

_output_shapes
:`:$	 

_output_shapes

: `: 


_output_shapes
:`
Ё
Р
/__inference_sequential_21_layer_call_fn_1930051
conv1d_8_input
unknown: 
	unknown_0: 
	unknown_1:	р@
	unknown_2:@
	unknown_3:@@
	unknown_4:@`
	unknown_5: `
	unknown_6:`
	unknown_7: 
	unknown_8:
	unknown_9:

unknown_10:
identityЂStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallconv1d_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_21_layer_call_and_return_conditional_losses_1930024o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameconv1d_8_input
ў
к

<__inference___backward_gpu_gru_with_fallback_1930324_1930460
placeholder
placeholder_1
placeholder_2
placeholder_33
/gradients_expanddims_1_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:џџџџџџџџџ d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:џџџџџџџџџ `
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:џџџџџџџџџ O
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 
!gradients/ExpandDims_1_grad/ShapeShape/gradients_expanddims_1_grad_shape_strided_slice*
T0*
_output_shapes
:Њ
#gradients/ExpandDims_1_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:Є
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Х
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_1_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:џџџџџџџџџ }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџ *
shrink_axis_maska
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*K
_output_shapes9
7:џџџџџџџџџ@:џџџџџџџџџ : :РI*
rnn_modegru
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:а
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:Х
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ \
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:g
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: 
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::ц
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:щ
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
: щ
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
: щ
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: щ
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: ь
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: ь
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   Ѓ
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   Ѕ
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   Ѕ
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ѕ
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ѕ
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ѕ
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  h
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ё
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ё
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ё
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ѓ
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Є
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Є
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: 
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:З
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ 
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:З
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ 
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:З
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ 
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:З
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:З
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:З
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Р
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:@`
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

: `m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   `   Ё
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:`r
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ e

Identity_2Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes

:@`g

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes

: `h

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes

:`"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ : :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ : ::џџџџџџџџџ@:џџџџџџџџџ : :РI::џџџџџџџџџ : ::::::: : : *<
api_implements*(gru_f821b5a3-099c-46c6-b85a-23e6bbf0193a*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_gru_with_fallback_1930459*
go_backwards( *

time_major( :- )
'
_output_shapes
:џџџџџџџџџ :1-
+
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :1-
+
_output_shapes
:џџџџџџџџџ :1-
+
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
::1	-
+
_output_shapes
:џџџџџџџџџ@:1
-
+
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :!

_output_shapes	
:РI: 

_output_shapes
::-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
>
К
'__forward_gpu_gru_with_fallback_1929980

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dimc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџV
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:РS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Џ
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    Ю
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*H
_output_shapes6
4:џџџџџџџџџ :џџџџџџџџџ : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:О
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџ *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_1
ExpandDimsstrided_slice_0:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:џџџџџџџџџ c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџ I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"

cudnnrnn_0CudnnRNN:output:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:џџџџџџџџџ@:џџџџџџџџџ :@`: `:`*<
api_implements*(gru_300f60c8-cd5f-48b0-871b-ea785a51c0be*
api_preferred_deviceGPU*X
backward_function_name><__inference___backward_gpu_gru_with_fallback_1929845_1929981*
go_backwards( *

time_major( :S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
п
Џ
while_cond_1932062
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1932062___redundant_placeholder05
1while_while_cond_1932062___redundant_placeholder15
1while_while_cond_1932062___redundant_placeholder25
1while_while_cond_1932062___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :џџџџџџџџџ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ@:

_output_shapes
: :

_output_shapes
:
	
ф
while_cond_1929677
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_1929677___redundant_placeholder05
1while_while_cond_1929677___redundant_placeholder15
1while_while_cond_1929677___redundant_placeholder25
1while_while_cond_1929677___redundant_placeholder35
1while_while_cond_1929677___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1: : : : :џџџџџџџџџ : :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
Щ4
А
)__inference_gpu_gru_with_fallback_1933870

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџV
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:РS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ъ
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:РIU
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    Ъ
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*H
_output_shapes6
4:џџџџџџџџџ :џџџџџџџџџ : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџ *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_1
ExpandDimsstrided_slice:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџ I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:џџџџџџџџџ@:џџџџџџџџџ :@`: `:`*<
api_implements*(gru_b015bfb0-6782-4f52-830c-ea8adfe5cd9c*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
и

*__inference_conv1d_8_layer_call_fn_1931937

inputs
unknown: 
	unknown_0: 
identityЂStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv1d_8_layer_call_and_return_conditional_losses_1929476s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
>
Є
 __inference_standard_gru_1931701

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3M
unstackUnpackbias*
T0* 
_output_shapes
:`:`*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*
shrink_axis_maskd
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:џџџџџџџџџ`h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ѓ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split^
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:џџџџџџџџџ`l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Љ
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ S
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:џџџџџџџџџ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ Q
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ў
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :џџџџџџџџџ : : :@`:`: `:`* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_1931611*
condR
while_cond_1931610*R
output_shapesA
?: : : : :џџџџџџџџџ : : :@`:`: `:`*
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ ]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:џџџџџџџџџ I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:џџџџџџџџџ@:џџџџџџџџџ :@`: `:`*<
api_implements*(gru_f1b5063f-a134-4d67-872c-0c565cc5da6e*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
	
ф
while_cond_1928048
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_1928048___redundant_placeholder05
1while_while_cond_1928048___redundant_placeholder15
1while_while_cond_1928048___redundant_placeholder25
1while_while_cond_1928048___redundant_placeholder35
1while_while_cond_1928048___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1: : : : :џџџџџџџџџ : :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
Є>
Є
 __inference_standard_gru_1932660

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3M
unstackUnpackbias*
T0* 
_output_shapes
:`:`*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*
shrink_axis_maskd
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:џџџџџџџџџ`h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ѓ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split^
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:џџџџџџџџџ`l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Љ
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ S
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:џџџџџџџџџ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ Q
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ў
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :џџџџџџџџџ : : :@`:`: `:`* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_1932570*
condR
while_cond_1932569*R
output_shapesA
?: : : : :џџџџџџџџџ : : :@`:`: `:`*
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ ]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:џџџџџџџџџ I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџ :@`: `:`*<
api_implements*(gru_8f43a4e9-8fbf-4cc8-8499-ccb41d598216*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
Ш	
і
E__inference_dense_29_layer_call_and_return_conditional_losses_1930017

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
	
ф
while_cond_1933325
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_1933325___redundant_placeholder05
1while_while_cond_1933325___redundant_placeholder15
1while_while_cond_1933325___redundant_placeholder25
1while_while_cond_1933325___redundant_placeholder35
1while_while_cond_1933325___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1: : : : :џџџџџџџџџ : :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
>
Є
 __inference_standard_gru_1933794

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3M
unstackUnpackbias*
T0* 
_output_shapes
:`:`*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*
shrink_axis_maskd
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:џџџџџџџџџ`h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ѓ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split^
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:џџџџџџџџџ`l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Љ
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ S
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:џџџџџџџџџ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ Q
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ў
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :џџџџџџџџџ : : :@`:`: `:`* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_1933704*
condR
while_cond_1933703*R
output_shapesA
?: : : : :џџџџџџџџџ : : :@`:`: `:`*
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ ]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:џџџџџџџџџ I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:џџџџџџџџџ@:џџџџџџџџџ :@`: `:`*<
api_implements*(gru_b015bfb0-6782-4f52-830c-ea8adfe5cd9c*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
э4
А
)__inference_gpu_gru_with_fallback_1929305

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџV
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:РS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ъ
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:РIU
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    г
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*Q
_output_shapes?
=:џџџџџџџџџџџџџџџџџџ :џџџџџџџџџ : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџ *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_1
ExpandDimsstrided_slice:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџ I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџ :@`: `:`*<
api_implements*(gru_7ffb1024-6f26-45b8-a31e-290afcdaacd2*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
4
 
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_1928664

inputs,
simple_rnn_cell_6_1928589:	р@'
simple_rnn_cell_6_1928591:@+
simple_rnn_cell_6_1928593:@@
identityЂ)simple_rnn_cell_6/StatefulPartitionedCallЂwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџрD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџр  р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџр*
shrink_axis_maskы
)simple_rnn_cell_6/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_6_1928589simple_rnn_cell_6_1928591simple_rnn_cell_6_1928593*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ@:џџџџџџџџџ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_simple_rnn_cell_6_layer_call_and_return_conditional_losses_1928549n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_6_1928589simple_rnn_cell_6_1928591simple_rnn_cell_6_1928593*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1928601*
condR
while_cond_1928600*8
output_shapes'
%: : : : :џџџџџџџџџ@: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   Ы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@z
NoOpNoOp*^simple_rnn_cell_6/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџџџџџџџџџџр: : : 2V
)simple_rnn_cell_6/StatefulPartitionedCall)simple_rnn_cell_6/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџр
 
_user_specified_nameinputs
я

Ж
%__inference_signature_wrapper_1930846
conv1d_8_input
unknown: 
	unknown_0: 
	unknown_1:	р@
	unknown_2:@
	unknown_3:@@
	unknown_4:@`
	unknown_5: `
	unknown_6:`
	unknown_7: 
	unknown_8:
	unknown_9:

unknown_10:
identityЂStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallconv1d_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_1928366o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameconv1d_8_input
>
К
'__forward_gpu_gru_with_fallback_1930459

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dimc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџV
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:РS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Џ
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    Ю
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*H
_output_shapes6
4:џџџџџџџџџ :џџџџџџџџџ : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:О
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџ *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_1
ExpandDimsstrided_slice_0:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:џџџџџџџџџ c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџ I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"

cudnnrnn_0CudnnRNN:output:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:џџџџџџџџџ@:џџџџџџџџџ :@`: `:`*<
api_implements*(gru_f821b5a3-099c-46c6-b85a-23e6bbf0193a*
api_preferred_deviceGPU*X
backward_function_name><__inference___backward_gpu_gru_with_fallback_1930324_1930460*
go_backwards( *

time_major( :S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
н-
у
while_body_1928049
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ@*
element_dtype0
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ`{
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:џџџџџџџџџ`W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Е
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ`
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:џџџџџџџџџ`Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Л
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_splitr
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ Y
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ t
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ ]
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ o
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ k
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ U

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ l
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ P
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?k
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ c
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ h
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/add_3:z:0*
_output_shapes
: *
element_dtype0:щшвO
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ "4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?: : : : :џџџџџџџџџ : : :@`:`: `:`: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@`: 

_output_shapes
:`:$	 

_output_shapes

: `: 


_output_shapes
:`

ъ
N__inference_simple_rnn_cell_6_layer_call_and_return_conditional_losses_1928549

inputs

states1
matmul_readvariableop_resource:	р@-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	р@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@G
TanhTanhadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:џџџџџџџџџр:џџџџџџџџџ@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџр
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_namestates
Ц
к

<__inference___backward_gpu_gru_with_fallback_1932737_1932873
placeholder
placeholder_1
placeholder_2
placeholder_33
/gradients_expanddims_1_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:џџџџџџџџџ d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:џџџџџџџџџ `
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:џџџџџџџџџ O
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 
!gradients/ExpandDims_1_grad/ShapeShape/gradients_expanddims_1_grad_shape_strided_slice*
T0*
_output_shapes
:Њ
#gradients/ExpandDims_1_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:Є
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Х
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_1_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:џџџџџџџџџ }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:Ђ
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *
shrink_axis_maska
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*T
_output_shapesB
@:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџ : :РI*
rnn_modegru
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:й
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:Х
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ \
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:g
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: 
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::ц
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:щ
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
: щ
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
: щ
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: щ
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: ь
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: ь
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   Ѓ
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   Ѕ
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   Ѕ
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ѕ
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ѕ
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ѕ
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  h
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ё
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ё
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ё
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ѓ
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Є
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Є
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: 
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:З
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ 
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:З
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ 
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:З
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ 
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:З
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:З
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:З
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Р
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:@`
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

: `m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   `   Ё
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:`{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ e

Identity_2Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes

:@`g

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes

: `h

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes

:`"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*Љ
_input_shapes
:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ : :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџџџџџџџџџџ : ::џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџ : :РI::џџџџџџџџџ : ::::::: : : *<
api_implements*(gru_8f43a4e9-8fbf-4cc8-8499-ccb41d598216*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_gru_with_fallback_1932872*
go_backwards( *

time_major( :- )
'
_output_shapes
:џџџџџџџџџ :1-
+
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :1-
+
_output_shapes
:џџџџџџџџџ ::6
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ :

_output_shapes
: :

_output_shapes
:::	6
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@:1
-
+
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :!

_output_shapes	
:РI: 

_output_shapes
::-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
л
Н
B__inference_gru_9_layer_call_and_return_conditional_losses_1934009

inputs.
read_readvariableop_resource:@`0
read_1_readvariableop_resource: `0
read_2_readvariableop_resource:`

identity_3ЂRead/ReadVariableOpЂRead_1/ReadVariableOpЂRead_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ p
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:@`*
dtype0Z
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:@`t
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

: `*
dtype0^

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

: `t
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes

:`*
dtype0^

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes

:`
PartitionedCallPartitionedCallinputszeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *S
_output_shapesA
?:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference_standard_gru_1933794i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ@: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
З>
К
'__forward_gpu_gru_with_fallback_1929052

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dimc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџV
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:РS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Џ
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    з
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*Q
_output_shapes?
=:џџџџџџџџџџџџџџџџџџ :џџџџџџџџџ : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:О
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџ *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_1
ExpandDimsstrided_slice_0:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:џџџџџџџџџ c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџ I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"

cudnnrnn_0CudnnRNN:output:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџ :@`: `:`*<
api_implements*(gru_e1cf211a-1d8c-47c6-ac5e-267262ea26df*
api_preferred_deviceGPU*X
backward_function_name><__inference___backward_gpu_gru_with_fallback_1928917_1929053*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias

З
'__inference_gru_9_layer_call_fn_1932464
inputs_0
unknown:@`
	unknown_0: `
	unknown_1:`
identityЂStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_gru_9_layer_call_and_return_conditional_losses_1929055o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
"
_user_specified_name
inputs/0
Ц
к

<__inference___backward_gpu_gru_with_fallback_1933115_1933251
placeholder
placeholder_1
placeholder_2
placeholder_33
/gradients_expanddims_1_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:џџџџџџџџџ d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:џџџџџџџџџ `
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:џџџџџџџџџ O
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 
!gradients/ExpandDims_1_grad/ShapeShape/gradients_expanddims_1_grad_shape_strided_slice*
T0*
_output_shapes
:Њ
#gradients/ExpandDims_1_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:Є
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Х
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_1_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:џџџџџџџџџ }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:Ђ
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *
shrink_axis_maska
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*T
_output_shapesB
@:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџ : :РI*
rnn_modegru
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:й
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:Х
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ \
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:g
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: 
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::ц
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:щ
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
: щ
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
: щ
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: щ
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: ь
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: ь
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   Ѓ
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   Ѕ
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   Ѕ
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ѕ
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ѕ
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ѕ
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  h
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ё
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ё
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ё
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ѓ
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Є
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Є
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: 
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:З
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ 
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:З
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ 
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:З
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ 
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:З
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:З
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:З
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Р
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:@`
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

: `m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   `   Ё
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:`{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ e

Identity_2Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes

:@`g

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes

: `h

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes

:`"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*Љ
_input_shapes
:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ : :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџџџџџџџџџџ : ::џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџ : :РI::џџџџџџџџџ : ::::::: : : *<
api_implements*(gru_d05ef4a6-d09c-40f3-878e-47b048895818*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_gru_with_fallback_1933250*
go_backwards( *

time_major( :- )
'
_output_shapes
:џџџџџџџџџ :1-
+
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :1-
+
_output_shapes
:џџџџџџџџџ ::6
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ :

_output_shapes
: :

_output_shapes
:::	6
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@:1
-
+
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :!

_output_shapes	
:РI: 

_output_shapes
::-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
щ=
П
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_1932237
inputs_0C
0simple_rnn_cell_6_matmul_readvariableop_resource:	р@?
1simple_rnn_cell_6_biasadd_readvariableop_resource:@D
2simple_rnn_cell_6_matmul_1_readvariableop_resource:@@
identityЂ(simple_rnn_cell_6/BiasAdd/ReadVariableOpЂ'simple_rnn_cell_6/MatMul/ReadVariableOpЂ)simple_rnn_cell_6/MatMul_1/ReadVariableOpЂwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџрD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџр  р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџр*
shrink_axis_mask
'simple_rnn_cell_6/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_6_matmul_readvariableop_resource*
_output_shapes
:	р@*
dtype0
simple_rnn_cell_6/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
(simple_rnn_cell_6/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ќ
simple_rnn_cell_6/BiasAddBiasAdd"simple_rnn_cell_6/MatMul:product:00simple_rnn_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
)simple_rnn_cell_6/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_6_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_6/MatMul_1MatMulzeros:output:01simple_rnn_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
simple_rnn_cell_6/addAddV2"simple_rnn_cell_6/BiasAdd:output:0$simple_rnn_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@k
simple_rnn_cell_6/TanhTanhsimple_rnn_cell_6/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : к
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_6_matmul_readvariableop_resource1simple_rnn_cell_6_biasadd_readvariableop_resource2simple_rnn_cell_6_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1932171*
condR
while_cond_1932170*8
output_shapes'
%: : : : :џџџџџџџџџ@: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   Ы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@Я
NoOpNoOp)^simple_rnn_cell_6/BiasAdd/ReadVariableOp(^simple_rnn_cell_6/MatMul/ReadVariableOp*^simple_rnn_cell_6/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџџџџџџџџџџр: : : 2T
(simple_rnn_cell_6/BiasAdd/ReadVariableOp(simple_rnn_cell_6/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_6/MatMul/ReadVariableOp'simple_rnn_cell_6/MatMul/ReadVariableOp2V
)simple_rnn_cell_6/MatMul_1/ReadVariableOp)simple_rnn_cell_6/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџр
"
_user_specified_name
inputs/0
З>
К
'__forward_gpu_gru_with_fallback_1933250

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dimc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџV
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:РS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Џ
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    з
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*Q
_output_shapes?
=:џџџџџџџџџџџџџџџџџџ :џџџџџџџџџ : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:О
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџ *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_1
ExpandDimsstrided_slice_0:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:џџџџџџџџџ c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџ I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"

cudnnrnn_0CudnnRNN:output:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџ :@`: `:`*<
api_implements*(gru_d05ef4a6-d09c-40f3-878e-47b048895818*
api_preferred_deviceGPU*X
backward_function_name><__inference___backward_gpu_gru_with_fallback_1933115_1933251*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
й
M
1__inference_repeat_vector_2_layer_call_fn_1931969

inputs
identityФ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_repeat_vector_2_layer_call_and_return_conditional_losses_1928378m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџџџџџџџџџџ:X T
0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ў
к

<__inference___backward_gpu_gru_with_fallback_1929845_1929981
placeholder
placeholder_1
placeholder_2
placeholder_33
/gradients_expanddims_1_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:џџџџџџџџџ d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:џџџџџџџџџ `
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:џџџџџџџџџ O
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 
!gradients/ExpandDims_1_grad/ShapeShape/gradients_expanddims_1_grad_shape_strided_slice*
T0*
_output_shapes
:Њ
#gradients/ExpandDims_1_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:Є
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Х
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_1_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:џџџџџџџџџ }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџ *
shrink_axis_maska
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*K
_output_shapes9
7:џџџџџџџџџ@:џџџџџџџџџ : :РI*
rnn_modegru
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:а
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:Х
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ \
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:g
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: 
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::ц
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:щ
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
: щ
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
: щ
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: щ
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: ь
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: ь
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   Ѓ
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   Ѕ
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   Ѕ
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ѕ
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ѕ
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ѕ
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  h
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ё
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ё
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ё
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ѓ
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Є
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Є
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: 
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:З
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ 
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:З
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ 
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:З
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ 
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:З
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:З
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:З
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Р
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:@`
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

: `m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   `   Ё
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:`r
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ e

Identity_2Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes

:@`g

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes

: `h

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes

:`"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ : :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ : ::џџџџџџџџџ@:џџџџџџџџџ : :РI::џџџџџџџџџ : ::::::: : : *<
api_implements*(gru_300f60c8-cd5f-48b0-871b-ea785a51c0be*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_gru_with_fallback_1929980*
go_backwards( *

time_major( :- )
'
_output_shapes
:џџџџџџџџџ :1-
+
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :1-
+
_output_shapes
:џџџџџџџџџ :1-
+
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
::1	-
+
_output_shapes
:џџџџџџџџџ@:1
-
+
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :!

_output_shapes	
:РI: 

_output_shapes
::-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Џ,
Ы
while_body_1932279
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
8while_simple_rnn_cell_6_matmul_readvariableop_resource_0:	р@G
9while_simple_rnn_cell_6_biasadd_readvariableop_resource_0:@L
:while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
6while_simple_rnn_cell_6_matmul_readvariableop_resource:	р@E
7while_simple_rnn_cell_6_biasadd_readvariableop_resource:@J
8while_simple_rnn_cell_6_matmul_1_readvariableop_resource:@@Ђ.while/simple_rnn_cell_6/BiasAdd/ReadVariableOpЂ-while/simple_rnn_cell_6/MatMul/ReadVariableOpЂ/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџр  Ї
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџр*
element_dtype0Ї
-while/simple_rnn_cell_6/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_6_matmul_readvariableop_resource_0*
_output_shapes
:	р@*
dtype0У
while/simple_rnn_cell_6/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@Є
.while/simple_rnn_cell_6/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_6_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0О
while/simple_rnn_cell_6/BiasAddBiasAdd(while/simple_rnn_cell_6/MatMul:product:06while/simple_rnn_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@Њ
/while/simple_rnn_cell_6/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0Њ
 while/simple_rnn_cell_6/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@Ќ
while/simple_rnn_cell_6/addAddV2(while/simple_rnn_cell_6/BiasAdd:output:0*while/simple_rnn_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@w
while/simple_rnn_cell_6/TanhTanhwhile/simple_rnn_cell_6/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Щ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_6/Tanh:y:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: }
while/Identity_4Identity while/simple_rnn_cell_6/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@п

while/NoOpNoOp/^while/simple_rnn_cell_6/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_6/MatMul/ReadVariableOp0^while/simple_rnn_cell_6/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_6_biasadd_readvariableop_resource9while_simple_rnn_cell_6_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_6_matmul_1_readvariableop_resource:while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_6_matmul_readvariableop_resource8while_simple_rnn_cell_6_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ@: : : : : 2`
.while/simple_rnn_cell_6/BiasAdd/ReadVariableOp.while/simple_rnn_cell_6/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_6/MatMul/ReadVariableOp-while/simple_rnn_cell_6/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ@:

_output_shapes
: :

_output_shapes
: 
Щ4
А
)__inference_gpu_gru_with_fallback_1933492

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџV
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:РS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ъ
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:РIU
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    Ъ
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*H
_output_shapes6
4:џџџџџџџџџ :џџџџџџџџџ : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџ *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_1
ExpandDimsstrided_slice:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџ I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:џџџџџџџџџ@:џџџџџџџџџ :@`: `:`*<
api_implements*(gru_351bbcc5-99cd-49ef-abe4-0838c20dc796*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
э4
А
)__inference_gpu_gru_with_fallback_1932736

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџV
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:РS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ъ
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:РIU
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    г
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*Q
_output_shapes?
=:џџџџџџџџџџџџџџџџџџ :џџџџџџџџџ : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџ *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_1
ExpandDimsstrided_slice:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџ I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџ :@`: `:`*<
api_implements*(gru_8f43a4e9-8fbf-4cc8-8499-ccb41d598216*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
М
h
L__inference_repeat_vector_2_layer_call_and_return_conditional_losses_1928378

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :x

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџZ
stackConst*
_output_shapes
:*
dtype0*!
valueB"         p
TileTileExpandDims:output:0stack:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџb
IdentityIdentityTile:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџџџџџџџџџџ:X T
0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
н-
у
while_body_1933326
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ@*
element_dtype0
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ`{
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:џџџџџџџџџ`W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Е
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ`
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:џџџџџџџџџ`Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Л
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_splitr
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ Y
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ t
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ ]
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ o
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ k
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ U

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ l
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ P
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?k
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ c
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ h
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/add_3:z:0*
_output_shapes
: *
element_dtype0:щшвO
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ "4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?: : : : :џџџџџџџџџ : : :@`:`: `:`: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@`: 

_output_shapes
:`:$	 

_output_shapes

: `: 


_output_shapes
:`
Р
b
F__inference_flatten_9_layer_call_and_return_conditional_losses_1929488

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџр  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџрY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџр"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ :S O
+
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

З
'__inference_gru_9_layer_call_fn_1932475
inputs_0
unknown:@`
	unknown_0: `
	unknown_1:`
identityЂStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_gru_9_layer_call_and_return_conditional_losses_1929444o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
"
_user_specified_name
inputs/0
э"
Щ
J__inference_sequential_21_layer_call_and_return_conditional_losses_1930809
conv1d_8_input&
conv1d_8_1930777: 
conv1d_8_1930779: '
simple_rnn_6_1930784:	р@"
simple_rnn_6_1930786:@&
simple_rnn_6_1930788:@@
gru_9_1930791:@`
gru_9_1930793: `
gru_9_1930795:`"
dense_28_1930798: 
dense_28_1930800:"
dense_29_1930803:
dense_29_1930805:
identityЂ conv1d_8/StatefulPartitionedCallЂ dense_28/StatefulPartitionedCallЂ dense_29/StatefulPartitionedCallЂgru_9/StatefulPartitionedCallЂ$simple_rnn_6/StatefulPartitionedCallџ
 conv1d_8/StatefulPartitionedCallStatefulPartitionedCallconv1d_8_inputconv1d_8_1930777conv1d_8_1930779*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv1d_8_layer_call_and_return_conditional_losses_1929476п
flatten_9/PartitionedCallPartitionedCall)conv1d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџр* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_9_layer_call_and_return_conditional_losses_1929488ш
repeat_vector_2/PartitionedCallPartitionedCall"flatten_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџр* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_repeat_vector_2_layer_call_and_return_conditional_losses_1928378С
$simple_rnn_6/StatefulPartitionedCallStatefulPartitionedCall(repeat_vector_2/PartitionedCall:output:0simple_rnn_6_1930784simple_rnn_6_1930786simple_rnn_6_1930788*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_1930592
gru_9/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_6/StatefulPartitionedCall:output:0gru_9_1930791gru_9_1930793gru_9_1930795*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_gru_9_layer_call_and_return_conditional_losses_1930462
 dense_28/StatefulPartitionedCallStatefulPartitionedCall&gru_9/StatefulPartitionedCall:output:0dense_28_1930798dense_28_1930800*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_28_layer_call_and_return_conditional_losses_1930001
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0dense_29_1930803dense_29_1930805*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_29_layer_call_and_return_conditional_losses_1930017x
IdentityIdentity)dense_29/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџі
NoOpNoOp!^conv1d_8/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall^gru_9/StatefulPartitionedCall%^simple_rnn_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ: : : : : : : : : : : : 2D
 conv1d_8/StatefulPartitionedCall conv1d_8/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2>
gru_9/StatefulPartitionedCallgru_9/StatefulPartitionedCall2L
$simple_rnn_6/StatefulPartitionedCall$simple_rnn_6/StatefulPartitionedCall:[ W
+
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameconv1d_8_input
Є>
Є
 __inference_standard_gru_1933038

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3M
unstackUnpackbias*
T0* 
_output_shapes
:`:`*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*
shrink_axis_maskd
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:џџџџџџџџџ`h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ѓ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split^
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:џџџџџџџџџ`l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Љ
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ S
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:џџџџџџџџџ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ Q
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ў
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :џџџџџџџџџ : : :@`:`: `:`* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_1932948*
condR
while_cond_1932947*R
output_shapesA
?: : : : :џџџџџџџџџ : : :@`:`: `:`*
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ ]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:џџџџџџџџџ I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџ :@`: `:`*<
api_implements*(gru_d05ef4a6-d09c-40f3-878e-47b048895818*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
э4
А
)__inference_gpu_gru_with_fallback_1933114

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџV
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:РS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ъ
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:РIU
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    г
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*Q
_output_shapes?
=:џџџџџџџџџџџџџџџџџџ :џџџџџџџџџ : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџ *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_1
ExpandDimsstrided_slice:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџ I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџ :@`: `:`*<
api_implements*(gru_d05ef4a6-d09c-40f3-878e-47b048895818*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
	
ф
while_cond_1932569
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_1932569___redundant_placeholder05
1while_while_cond_1932569___redundant_placeholder15
1while_while_cond_1932569___redundant_placeholder25
1while_while_cond_1932569___redundant_placeholder35
1while_while_cond_1932569___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1: : : : :џџџџџџџџџ : :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:

ь
N__inference_simple_rnn_cell_6_layer_call_and_return_conditional_losses_1934109

inputs
states_01
matmul_readvariableop_resource:	р@-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	р@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@G
TanhTanhadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:џџџџџџџџџр:џџџџџџџџџ@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџр
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ@
"
_user_specified_name
states/0
щ=
П
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_1932129
inputs_0C
0simple_rnn_cell_6_matmul_readvariableop_resource:	р@?
1simple_rnn_cell_6_biasadd_readvariableop_resource:@D
2simple_rnn_cell_6_matmul_1_readvariableop_resource:@@
identityЂ(simple_rnn_cell_6/BiasAdd/ReadVariableOpЂ'simple_rnn_cell_6/MatMul/ReadVariableOpЂ)simple_rnn_cell_6/MatMul_1/ReadVariableOpЂwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџрD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџр  р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџр*
shrink_axis_mask
'simple_rnn_cell_6/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_6_matmul_readvariableop_resource*
_output_shapes
:	р@*
dtype0
simple_rnn_cell_6/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
(simple_rnn_cell_6/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ќ
simple_rnn_cell_6/BiasAddBiasAdd"simple_rnn_cell_6/MatMul:product:00simple_rnn_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
)simple_rnn_cell_6/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_6_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_6/MatMul_1MatMulzeros:output:01simple_rnn_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
simple_rnn_cell_6/addAddV2"simple_rnn_cell_6/BiasAdd:output:0$simple_rnn_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@k
simple_rnn_cell_6/TanhTanhsimple_rnn_cell_6/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : к
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_6_matmul_readvariableop_resource1simple_rnn_cell_6_biasadd_readvariableop_resource2simple_rnn_cell_6_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1932063*
condR
while_cond_1932062*8
output_shapes'
%: : : : :џџџџџџџџџ@: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   Ы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@Я
NoOpNoOp)^simple_rnn_cell_6/BiasAdd/ReadVariableOp(^simple_rnn_cell_6/MatMul/ReadVariableOp*^simple_rnn_cell_6/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџџџџџџџџџџр: : : 2T
(simple_rnn_cell_6/BiasAdd/ReadVariableOp(simple_rnn_cell_6/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_6/MatMul/ReadVariableOp'simple_rnn_cell_6/MatMul/ReadVariableOp2V
)simple_rnn_cell_6/MatMul_1/ReadVariableOp)simple_rnn_cell_6/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџр
"
_user_specified_name
inputs/0
Щ4
А
)__inference_gpu_gru_with_fallback_1930323

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџV
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:РS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ъ
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:РIU
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    Ъ
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*H
_output_shapes6
4:џџџџџџџџџ :џџџџџџџџџ : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџ *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_1
ExpandDimsstrided_slice:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџ I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:џџџџџџџџџ@:џџџџџџџџџ :@`: `:`*<
api_implements*(gru_f821b5a3-099c-46c6-b85a-23e6bbf0193a*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
>
Є
 __inference_standard_gru_1928139

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3M
unstackUnpackbias*
T0* 
_output_shapes
:`:`*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*
shrink_axis_maskd
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:џџџџџџџџџ`h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ѓ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split^
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:џџџџџџџџџ`l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Љ
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ S
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:џџџџџџџџџ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ Q
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ў
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :џџџџџџџџџ : : :@`:`: `:`* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_1928049*
condR
while_cond_1928048*R
output_shapesA
?: : : : :џџџџџџџџџ : : :@`:`: `:`*
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ ]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:џџџџџџџџџ I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:џџџџџџџџџ@:џџџџџџџџџ :@`: `:`*<
api_implements*(gru_73e98d14-b186-4092-be4c-44eb47643dc0*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
Щ

E__inference_conv1d_8_layer_call_and_return_conditional_losses_1929476

inputsA
+conv1d_expanddims_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂ"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: ­
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџ *
squeeze_dims

§џџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ 
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
З>
К
'__forward_gpu_gru_with_fallback_1932872

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dimc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџV
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:РS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Џ
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    з
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*Q
_output_shapes?
=:џџџџџџџџџџџџџџџџџџ :џџџџџџџџџ : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:О
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџ *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_1
ExpandDimsstrided_slice_0:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:џџџџџџџџџ c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџ I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"

cudnnrnn_0CudnnRNN:output:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџ :@`: `:`*<
api_implements*(gru_8f43a4e9-8fbf-4cc8-8499-ccb41d598216*
api_preferred_deviceGPU*X
backward_function_name><__inference___backward_gpu_gru_with_fallback_1932737_1932873*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
К
А
-sequential_21_simple_rnn_6_while_cond_1927913R
Nsequential_21_simple_rnn_6_while_sequential_21_simple_rnn_6_while_loop_counterX
Tsequential_21_simple_rnn_6_while_sequential_21_simple_rnn_6_while_maximum_iterations0
,sequential_21_simple_rnn_6_while_placeholder2
.sequential_21_simple_rnn_6_while_placeholder_12
.sequential_21_simple_rnn_6_while_placeholder_2T
Psequential_21_simple_rnn_6_while_less_sequential_21_simple_rnn_6_strided_slice_1k
gsequential_21_simple_rnn_6_while_sequential_21_simple_rnn_6_while_cond_1927913___redundant_placeholder0k
gsequential_21_simple_rnn_6_while_sequential_21_simple_rnn_6_while_cond_1927913___redundant_placeholder1k
gsequential_21_simple_rnn_6_while_sequential_21_simple_rnn_6_while_cond_1927913___redundant_placeholder2k
gsequential_21_simple_rnn_6_while_sequential_21_simple_rnn_6_while_cond_1927913___redundant_placeholder3-
)sequential_21_simple_rnn_6_while_identity
Ю
%sequential_21/simple_rnn_6/while/LessLess,sequential_21_simple_rnn_6_while_placeholderPsequential_21_simple_rnn_6_while_less_sequential_21_simple_rnn_6_strided_slice_1*
T0*
_output_shapes
: 
)sequential_21/simple_rnn_6/while/IdentityIdentity)sequential_21/simple_rnn_6/while/Less:z:0*
T0
*
_output_shapes
: "_
)sequential_21_simple_rnn_6_while_identity2sequential_21/simple_rnn_6/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :џџџџџџџџџ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ@:

_output_shapes
: :

_output_shapes
:
Е
Л
.__inference_simple_rnn_6_layer_call_fn_1931988
inputs_0
unknown:	р@
	unknown_0:@
	unknown_1:@@
identityЂStatefulPartitionedCallњ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_1928505|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџџџџџџџџџџр: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџр
"
_user_specified_name
inputs/0
бE
љ
-sequential_21_simple_rnn_6_while_body_1927914R
Nsequential_21_simple_rnn_6_while_sequential_21_simple_rnn_6_while_loop_counterX
Tsequential_21_simple_rnn_6_while_sequential_21_simple_rnn_6_while_maximum_iterations0
,sequential_21_simple_rnn_6_while_placeholder2
.sequential_21_simple_rnn_6_while_placeholder_12
.sequential_21_simple_rnn_6_while_placeholder_2Q
Msequential_21_simple_rnn_6_while_sequential_21_simple_rnn_6_strided_slice_1_0
sequential_21_simple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_sequential_21_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor_0f
Ssequential_21_simple_rnn_6_while_simple_rnn_cell_6_matmul_readvariableop_resource_0:	р@b
Tsequential_21_simple_rnn_6_while_simple_rnn_cell_6_biasadd_readvariableop_resource_0:@g
Usequential_21_simple_rnn_6_while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0:@@-
)sequential_21_simple_rnn_6_while_identity/
+sequential_21_simple_rnn_6_while_identity_1/
+sequential_21_simple_rnn_6_while_identity_2/
+sequential_21_simple_rnn_6_while_identity_3/
+sequential_21_simple_rnn_6_while_identity_4O
Ksequential_21_simple_rnn_6_while_sequential_21_simple_rnn_6_strided_slice_1
sequential_21_simple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_sequential_21_simple_rnn_6_tensorarrayunstack_tensorlistfromtensord
Qsequential_21_simple_rnn_6_while_simple_rnn_cell_6_matmul_readvariableop_resource:	р@`
Rsequential_21_simple_rnn_6_while_simple_rnn_cell_6_biasadd_readvariableop_resource:@e
Ssequential_21_simple_rnn_6_while_simple_rnn_cell_6_matmul_1_readvariableop_resource:@@ЂIsequential_21/simple_rnn_6/while/simple_rnn_cell_6/BiasAdd/ReadVariableOpЂHsequential_21/simple_rnn_6/while/simple_rnn_cell_6/MatMul/ReadVariableOpЂJsequential_21/simple_rnn_6/while/simple_rnn_cell_6/MatMul_1/ReadVariableOpЃ
Rsequential_21/simple_rnn_6/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџр  Џ
Dsequential_21/simple_rnn_6/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_21_simple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_sequential_21_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor_0,sequential_21_simple_rnn_6_while_placeholder[sequential_21/simple_rnn_6/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџр*
element_dtype0н
Hsequential_21/simple_rnn_6/while/simple_rnn_cell_6/MatMul/ReadVariableOpReadVariableOpSsequential_21_simple_rnn_6_while_simple_rnn_cell_6_matmul_readvariableop_resource_0*
_output_shapes
:	р@*
dtype0
9sequential_21/simple_rnn_6/while/simple_rnn_cell_6/MatMulMatMulKsequential_21/simple_rnn_6/while/TensorArrayV2Read/TensorListGetItem:item:0Psequential_21/simple_rnn_6/while/simple_rnn_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@к
Isequential_21/simple_rnn_6/while/simple_rnn_cell_6/BiasAdd/ReadVariableOpReadVariableOpTsequential_21_simple_rnn_6_while_simple_rnn_cell_6_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0
:sequential_21/simple_rnn_6/while/simple_rnn_cell_6/BiasAddBiasAddCsequential_21/simple_rnn_6/while/simple_rnn_cell_6/MatMul:product:0Qsequential_21/simple_rnn_6/while/simple_rnn_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@р
Jsequential_21/simple_rnn_6/while/simple_rnn_cell_6/MatMul_1/ReadVariableOpReadVariableOpUsequential_21_simple_rnn_6_while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0ћ
;sequential_21/simple_rnn_6/while/simple_rnn_cell_6/MatMul_1MatMul.sequential_21_simple_rnn_6_while_placeholder_2Rsequential_21/simple_rnn_6/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@§
6sequential_21/simple_rnn_6/while/simple_rnn_cell_6/addAddV2Csequential_21/simple_rnn_6/while/simple_rnn_cell_6/BiasAdd:output:0Esequential_21/simple_rnn_6/while/simple_rnn_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@­
7sequential_21/simple_rnn_6/while/simple_rnn_cell_6/TanhTanh:sequential_21/simple_rnn_6/while/simple_rnn_cell_6/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Е
Esequential_21/simple_rnn_6/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem.sequential_21_simple_rnn_6_while_placeholder_1,sequential_21_simple_rnn_6_while_placeholder;sequential_21/simple_rnn_6/while/simple_rnn_cell_6/Tanh:y:0*
_output_shapes
: *
element_dtype0:щшвh
&sequential_21/simple_rnn_6/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :­
$sequential_21/simple_rnn_6/while/addAddV2,sequential_21_simple_rnn_6_while_placeholder/sequential_21/simple_rnn_6/while/add/y:output:0*
T0*
_output_shapes
: j
(sequential_21/simple_rnn_6/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :г
&sequential_21/simple_rnn_6/while/add_1AddV2Nsequential_21_simple_rnn_6_while_sequential_21_simple_rnn_6_while_loop_counter1sequential_21/simple_rnn_6/while/add_1/y:output:0*
T0*
_output_shapes
: Њ
)sequential_21/simple_rnn_6/while/IdentityIdentity*sequential_21/simple_rnn_6/while/add_1:z:0&^sequential_21/simple_rnn_6/while/NoOp*
T0*
_output_shapes
: ж
+sequential_21/simple_rnn_6/while/Identity_1IdentityTsequential_21_simple_rnn_6_while_sequential_21_simple_rnn_6_while_maximum_iterations&^sequential_21/simple_rnn_6/while/NoOp*
T0*
_output_shapes
: Њ
+sequential_21/simple_rnn_6/while/Identity_2Identity(sequential_21/simple_rnn_6/while/add:z:0&^sequential_21/simple_rnn_6/while/NoOp*
T0*
_output_shapes
: з
+sequential_21/simple_rnn_6/while/Identity_3IdentityUsequential_21/simple_rnn_6/while/TensorArrayV2Write/TensorListSetItem:output_handle:0&^sequential_21/simple_rnn_6/while/NoOp*
T0*
_output_shapes
: Ю
+sequential_21/simple_rnn_6/while/Identity_4Identity;sequential_21/simple_rnn_6/while/simple_rnn_cell_6/Tanh:y:0&^sequential_21/simple_rnn_6/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@Ы
%sequential_21/simple_rnn_6/while/NoOpNoOpJ^sequential_21/simple_rnn_6/while/simple_rnn_cell_6/BiasAdd/ReadVariableOpI^sequential_21/simple_rnn_6/while/simple_rnn_cell_6/MatMul/ReadVariableOpK^sequential_21/simple_rnn_6/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "_
)sequential_21_simple_rnn_6_while_identity2sequential_21/simple_rnn_6/while/Identity:output:0"c
+sequential_21_simple_rnn_6_while_identity_14sequential_21/simple_rnn_6/while/Identity_1:output:0"c
+sequential_21_simple_rnn_6_while_identity_24sequential_21/simple_rnn_6/while/Identity_2:output:0"c
+sequential_21_simple_rnn_6_while_identity_34sequential_21/simple_rnn_6/while/Identity_3:output:0"c
+sequential_21_simple_rnn_6_while_identity_44sequential_21/simple_rnn_6/while/Identity_4:output:0"
Ksequential_21_simple_rnn_6_while_sequential_21_simple_rnn_6_strided_slice_1Msequential_21_simple_rnn_6_while_sequential_21_simple_rnn_6_strided_slice_1_0"Њ
Rsequential_21_simple_rnn_6_while_simple_rnn_cell_6_biasadd_readvariableop_resourceTsequential_21_simple_rnn_6_while_simple_rnn_cell_6_biasadd_readvariableop_resource_0"Ќ
Ssequential_21_simple_rnn_6_while_simple_rnn_cell_6_matmul_1_readvariableop_resourceUsequential_21_simple_rnn_6_while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0"Ј
Qsequential_21_simple_rnn_6_while_simple_rnn_cell_6_matmul_readvariableop_resourceSsequential_21_simple_rnn_6_while_simple_rnn_cell_6_matmul_readvariableop_resource_0"
sequential_21_simple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_sequential_21_simple_rnn_6_tensorarrayunstack_tensorlistfromtensorsequential_21_simple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_sequential_21_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ@: : : : : 2
Isequential_21/simple_rnn_6/while/simple_rnn_cell_6/BiasAdd/ReadVariableOpIsequential_21/simple_rnn_6/while/simple_rnn_cell_6/BiasAdd/ReadVariableOp2
Hsequential_21/simple_rnn_6/while/simple_rnn_cell_6/MatMul/ReadVariableOpHsequential_21/simple_rnn_6/while/simple_rnn_cell_6/MatMul/ReadVariableOp2
Jsequential_21/simple_rnn_6/while/simple_rnn_cell_6/MatMul_1/ReadVariableOpJsequential_21/simple_rnn_6/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ@:

_output_shapes
: :

_output_shapes
: 
Щ4
А
)__inference_gpu_gru_with_fallback_1929844

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџV
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:РS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ъ
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:РIU
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    Ъ
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*H
_output_shapes6
4:џџџџџџџџџ :џџџџџџџџџ : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџ *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_1
ExpandDimsstrided_slice:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџ I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:џџџџџџџџџ@:џџџџџџџџџ :@`: `:`*<
api_implements*(gru_300f60c8-cd5f-48b0-871b-ea785a51c0be*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
ў
к

<__inference___backward_gpu_gru_with_fallback_1933493_1933629
placeholder
placeholder_1
placeholder_2
placeholder_33
/gradients_expanddims_1_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:џџџџџџџџџ d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:џџџџџџџџџ `
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:џџџџџџџџџ O
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 
!gradients/ExpandDims_1_grad/ShapeShape/gradients_expanddims_1_grad_shape_strided_slice*
T0*
_output_shapes
:Њ
#gradients/ExpandDims_1_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:Є
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Х
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_1_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:џџџџџџџџџ }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџ *
shrink_axis_maska
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*K
_output_shapes9
7:џџџџџџџџџ@:џџџџџџџџџ : :РI*
rnn_modegru
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:а
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:Х
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ \
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:g
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: 
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::ц
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:щ
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
: щ
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
: щ
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: щ
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: ь
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: ь
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   Ѓ
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   Ѕ
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   Ѕ
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ѕ
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ѕ
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ѕ
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  h
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ё
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ё
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ё
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ѓ
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Є
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Є
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: 
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:З
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ 
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:З
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ 
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:З
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ 
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:З
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:З
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:З
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Р
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:@`
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

: `m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   `   Ё
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:`r
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ e

Identity_2Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes

:@`g

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes

: `h

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes

:`"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ : :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ : ::џџџџџџџџџ@:џџџџџџџџџ : :РI::џџџџџџџџџ : ::::::: : : *<
api_implements*(gru_351bbcc5-99cd-49ef-abe4-0838c20dc796*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_gru_with_fallback_1933628*
go_backwards( *

time_major( :- )
'
_output_shapes
:џџџџџџџџџ :1-
+
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :1-
+
_output_shapes
:џџџџџџџџџ :1-
+
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
::1	-
+
_output_shapes
:џџџџџџџџџ@:1
-
+
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :!

_output_shapes	
:РI: 

_output_shapes
::-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
э
Н
B__inference_gru_9_layer_call_and_return_conditional_losses_1929055

inputs.
read_readvariableop_resource:@`0
read_1_readvariableop_resource: `0
read_2_readvariableop_resource:`

identity_3ЂRead/ReadVariableOpЂRead_1/ReadVariableOpЂRead_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ p
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:@`*
dtype0Z
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:@`t
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

: `*
dtype0^

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

: `t
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes

:`*
dtype0^

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes

:`
PartitionedCallPartitionedCallinputszeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *S
_output_shapesA
?:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference_standard_gru_1928840i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ@: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
н-
у
while_body_1930157
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ@*
element_dtype0
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ`{
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:џџџџџџџџџ`W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Е
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ`
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:џџџџџџџџџ`Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Л
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_splitr
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ Y
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ t
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ ]
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ o
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ k
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ U

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ l
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ P
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?k
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ c
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ h
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/add_3:z:0*
_output_shapes
: *
element_dtype0:щшвO
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ "4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?: : : : :џџџџџџџџџ : : :@`:`: `:`: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@`: 

_output_shapes
:`:$	 

_output_shapes

: `: 


_output_shapes
:`
!
к
while_body_1928601
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
!while_simple_rnn_cell_6_1928623_0:	р@/
!while_simple_rnn_cell_6_1928625_0:@3
!while_simple_rnn_cell_6_1928627_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
while_simple_rnn_cell_6_1928623:	р@-
while_simple_rnn_cell_6_1928625:@1
while_simple_rnn_cell_6_1928627:@@Ђ/while/simple_rnn_cell_6/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџр  Ї
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџр*
element_dtype0І
/while/simple_rnn_cell_6/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2!while_simple_rnn_cell_6_1928623_0!while_simple_rnn_cell_6_1928625_0!while_simple_rnn_cell_6_1928627_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ@:џџџџџџџџџ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_simple_rnn_cell_6_layer_call_and_return_conditional_losses_1928549с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder8while/simple_rnn_cell_6/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity8while/simple_rnn_cell_6/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@~

while/NoOpNoOp0^while/simple_rnn_cell_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_simple_rnn_cell_6_1928623!while_simple_rnn_cell_6_1928623_0"D
while_simple_rnn_cell_6_1928625!while_simple_rnn_cell_6_1928625_0"D
while_simple_rnn_cell_6_1928627!while_simple_rnn_cell_6_1928627_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ@: : : : : 2b
/while/simple_rnn_cell_6/StatefulPartitionedCall/while/simple_rnn_cell_6/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ@:

_output_shapes
: :

_output_shapes
: 
ѕ
П
B__inference_gru_9_layer_call_and_return_conditional_losses_1933253
inputs_0.
read_readvariableop_resource:@`0
read_1_readvariableop_resource: `0
read_2_readvariableop_resource:`

identity_3ЂRead/ReadVariableOpЂRead_1/ReadVariableOpЂRead_2/ReadVariableOp=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ p
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:@`*
dtype0Z
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:@`t
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

: `*
dtype0^

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

: `t
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes

:`*
dtype0^

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes

:`
PartitionedCallPartitionedCallinputs_0zeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *S
_output_shapesA
?:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference_standard_gru_1933038i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ@: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
"
_user_specified_name
inputs/0
п
Џ
while_cond_1930525
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1930525___redundant_placeholder05
1while_while_cond_1930525___redundant_placeholder15
1while_while_cond_1930525___redundant_placeholder25
1while_while_cond_1930525___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :џџџџџџџџџ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ@:

_output_shapes
: :

_output_shapes
:
л
Н
B__inference_gru_9_layer_call_and_return_conditional_losses_1930462

inputs.
read_readvariableop_resource:@`0
read_1_readvariableop_resource: `0
read_2_readvariableop_resource:`

identity_3ЂRead/ReadVariableOpЂRead_1/ReadVariableOpЂRead_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ p
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:@`*
dtype0Z
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:@`t
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

: `*
dtype0^

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

: `t
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes

:`*
dtype0^

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes

:`
PartitionedCallPartitionedCallinputszeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *S
_output_shapesA
?:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference_standard_gru_1930247i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ@: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
>
К
'__forward_gpu_gru_with_fallback_1934006

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dimc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџV
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:РS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Џ
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    Ю
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*H
_output_shapes6
4:џџџџџџџџџ :џџџџџџџџџ : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:О
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџ *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_1
ExpandDimsstrided_slice_0:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:џџџџџџџџџ c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџ I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"

cudnnrnn_0CudnnRNN:output:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:џџџџџџџџџ@:џџџџџџџџџ :@`: `:`*<
api_implements*(gru_b015bfb0-6782-4f52-830c-ea8adfe5cd9c*
api_preferred_deviceGPU*X
backward_function_name><__inference___backward_gpu_gru_with_fallback_1933871_1934007*
go_backwards( *

time_major( :S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
п
Џ
while_cond_1928600
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1928600___redundant_placeholder05
1while_while_cond_1928600___redundant_placeholder15
1while_while_cond_1928600___redundant_placeholder25
1while_while_cond_1928600___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :џџџџџџџџџ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ@:

_output_shapes
: :

_output_shapes
:
	
ф
while_cond_1932947
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_1932947___redundant_placeholder05
1while_while_cond_1932947___redundant_placeholder15
1while_while_cond_1932947___redundant_placeholder25
1while_while_cond_1932947___redundant_placeholder35
1while_while_cond_1932947___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1: : : : :џџџџџџџџџ : :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
О

м
3__inference_simple_rnn_cell_6_layer_call_fn_1934075

inputs
states_0
unknown:	р@
	unknown_0:@
	unknown_1:@@
identity

identity_1ЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ@:џџџџџџџџџ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_simple_rnn_cell_6_layer_call_and_return_conditional_losses_1928549o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:џџџџџџџџџр:џџџџџџџџџ@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџр
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ@
"
_user_specified_name
states/0
э4
А
)__inference_gpu_gru_with_fallback_1928916

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџV
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:РS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ъ
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:РIU
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    г
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*Q
_output_shapes?
=:џџџџџџџџџџџџџџџџџџ :џџџџџџџџџ : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџ *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_1
ExpandDimsstrided_slice:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџ I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџ :@`: `:`*<
api_implements*(gru_e1cf211a-1d8c-47c6-ac5e-267262ea26df*
api_preferred_deviceGPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
Р
b
F__inference_flatten_9_layer_call_and_return_conditional_losses_1931964

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџр  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџрY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџр"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ :S O
+
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
П

І
simple_rnn_6_while_cond_19314756
2simple_rnn_6_while_simple_rnn_6_while_loop_counter<
8simple_rnn_6_while_simple_rnn_6_while_maximum_iterations"
simple_rnn_6_while_placeholder$
 simple_rnn_6_while_placeholder_1$
 simple_rnn_6_while_placeholder_28
4simple_rnn_6_while_less_simple_rnn_6_strided_slice_1O
Ksimple_rnn_6_while_simple_rnn_6_while_cond_1931475___redundant_placeholder0O
Ksimple_rnn_6_while_simple_rnn_6_while_cond_1931475___redundant_placeholder1O
Ksimple_rnn_6_while_simple_rnn_6_while_cond_1931475___redundant_placeholder2O
Ksimple_rnn_6_while_simple_rnn_6_while_cond_1931475___redundant_placeholder3
simple_rnn_6_while_identity

simple_rnn_6/while/LessLesssimple_rnn_6_while_placeholder4simple_rnn_6_while_less_simple_rnn_6_strided_slice_1*
T0*
_output_shapes
: e
simple_rnn_6/while/IdentityIdentitysimple_rnn_6/while/Less:z:0*
T0
*
_output_shapes
: "C
simple_rnn_6_while_identity$simple_rnn_6/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :џџџџџџџџџ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ@:

_output_shapes
: :

_output_shapes
:
>
Є
 __inference_standard_gru_1929768

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3M
unstackUnpackbias*
T0* 
_output_shapes
:`:`*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*
shrink_axis_maskd
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:џџџџџџџџџ`h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ѓ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split^
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:џџџџџџџџџ`l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Љ
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ S
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:џџџџџџџџџ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ Q
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ў
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :џџџџџџџџџ : : :@`:`: `:`* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_1929678*
condR
while_cond_1929677*R
output_shapesA
?: : : : :џџџџџџџџџ : : :@`:`: `:`*
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ ]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:џџџџџџџџџ I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:џџџџџџџџџ@:џџџџџџџџџ :@`: `:`*<
api_implements*(gru_300f60c8-cd5f-48b0-871b-ea785a51c0be*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
и

"__inference__wrapped_model_1928366
conv1d_8_inputX
Bsequential_21_conv1d_8_conv1d_expanddims_1_readvariableop_resource: D
6sequential_21_conv1d_8_biasadd_readvariableop_resource: ^
Ksequential_21_simple_rnn_6_simple_rnn_cell_6_matmul_readvariableop_resource:	р@Z
Lsequential_21_simple_rnn_6_simple_rnn_cell_6_biasadd_readvariableop_resource:@_
Msequential_21_simple_rnn_6_simple_rnn_cell_6_matmul_1_readvariableop_resource:@@B
0sequential_21_gru_9_read_readvariableop_resource:@`D
2sequential_21_gru_9_read_1_readvariableop_resource: `D
2sequential_21_gru_9_read_2_readvariableop_resource:`G
5sequential_21_dense_28_matmul_readvariableop_resource: D
6sequential_21_dense_28_biasadd_readvariableop_resource:G
5sequential_21_dense_29_matmul_readvariableop_resource:D
6sequential_21_dense_29_biasadd_readvariableop_resource:
identityЂ-sequential_21/conv1d_8/BiasAdd/ReadVariableOpЂ9sequential_21/conv1d_8/Conv1D/ExpandDims_1/ReadVariableOpЂ-sequential_21/dense_28/BiasAdd/ReadVariableOpЂ,sequential_21/dense_28/MatMul/ReadVariableOpЂ-sequential_21/dense_29/BiasAdd/ReadVariableOpЂ,sequential_21/dense_29/MatMul/ReadVariableOpЂ'sequential_21/gru_9/Read/ReadVariableOpЂ)sequential_21/gru_9/Read_1/ReadVariableOpЂ)sequential_21/gru_9/Read_2/ReadVariableOpЂCsequential_21/simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOpЂBsequential_21/simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOpЂDsequential_21/simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOpЂ sequential_21/simple_rnn_6/whilew
,sequential_21/conv1d_8/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџЗ
(sequential_21/conv1d_8/Conv1D/ExpandDims
ExpandDimsconv1d_8_input5sequential_21/conv1d_8/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџР
9sequential_21/conv1d_8/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpBsequential_21_conv1d_8_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0p
.sequential_21/conv1d_8/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : х
*sequential_21/conv1d_8/Conv1D/ExpandDims_1
ExpandDimsAsequential_21/conv1d_8/Conv1D/ExpandDims_1/ReadVariableOp:value:07sequential_21/conv1d_8/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: ђ
sequential_21/conv1d_8/Conv1DConv2D1sequential_21/conv1d_8/Conv1D/ExpandDims:output:03sequential_21/conv1d_8/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
Ў
%sequential_21/conv1d_8/Conv1D/SqueezeSqueeze&sequential_21/conv1d_8/Conv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџ *
squeeze_dims

§џџџџџџџџ 
-sequential_21/conv1d_8/BiasAdd/ReadVariableOpReadVariableOp6sequential_21_conv1d_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ц
sequential_21/conv1d_8/BiasAddBiasAdd.sequential_21/conv1d_8/Conv1D/Squeeze:output:05sequential_21/conv1d_8/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ 
sequential_21/conv1d_8/ReluRelu'sequential_21/conv1d_8/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ n
sequential_21/flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџр  А
sequential_21/flatten_9/ReshapeReshape)sequential_21/conv1d_8/Relu:activations:0&sequential_21/flatten_9/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџрn
,sequential_21/repeat_vector_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Ю
(sequential_21/repeat_vector_2/ExpandDims
ExpandDims(sequential_21/flatten_9/Reshape:output:05sequential_21/repeat_vector_2/ExpandDims/dim:output:0*
T0*,
_output_shapes
:џџџџџџџџџрx
#sequential_21/repeat_vector_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"         Т
"sequential_21/repeat_vector_2/TileTile1sequential_21/repeat_vector_2/ExpandDims:output:0,sequential_21/repeat_vector_2/stack:output:0*
T0*,
_output_shapes
:џџџџџџџџџр{
 sequential_21/simple_rnn_6/ShapeShape+sequential_21/repeat_vector_2/Tile:output:0*
T0*
_output_shapes
:x
.sequential_21/simple_rnn_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0sequential_21/simple_rnn_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0sequential_21/simple_rnn_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:и
(sequential_21/simple_rnn_6/strided_sliceStridedSlice)sequential_21/simple_rnn_6/Shape:output:07sequential_21/simple_rnn_6/strided_slice/stack:output:09sequential_21/simple_rnn_6/strided_slice/stack_1:output:09sequential_21/simple_rnn_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)sequential_21/simple_rnn_6/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@Ф
'sequential_21/simple_rnn_6/zeros/packedPack1sequential_21/simple_rnn_6/strided_slice:output:02sequential_21/simple_rnn_6/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:k
&sequential_21/simple_rnn_6/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Н
 sequential_21/simple_rnn_6/zerosFill0sequential_21/simple_rnn_6/zeros/packed:output:0/sequential_21/simple_rnn_6/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@~
)sequential_21/simple_rnn_6/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Щ
$sequential_21/simple_rnn_6/transpose	Transpose+sequential_21/repeat_vector_2/Tile:output:02sequential_21/simple_rnn_6/transpose/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџрz
"sequential_21/simple_rnn_6/Shape_1Shape(sequential_21/simple_rnn_6/transpose:y:0*
T0*
_output_shapes
:z
0sequential_21/simple_rnn_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2sequential_21/simple_rnn_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2sequential_21/simple_rnn_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:т
*sequential_21/simple_rnn_6/strided_slice_1StridedSlice+sequential_21/simple_rnn_6/Shape_1:output:09sequential_21/simple_rnn_6/strided_slice_1/stack:output:0;sequential_21/simple_rnn_6/strided_slice_1/stack_1:output:0;sequential_21/simple_rnn_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
6sequential_21/simple_rnn_6/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
(sequential_21/simple_rnn_6/TensorArrayV2TensorListReserve?sequential_21/simple_rnn_6/TensorArrayV2/element_shape:output:03sequential_21/simple_rnn_6/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвЁ
Psequential_21/simple_rnn_6/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџр  Б
Bsequential_21/simple_rnn_6/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor(sequential_21/simple_rnn_6/transpose:y:0Ysequential_21/simple_rnn_6/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвz
0sequential_21/simple_rnn_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2sequential_21/simple_rnn_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2sequential_21/simple_rnn_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ё
*sequential_21/simple_rnn_6/strided_slice_2StridedSlice(sequential_21/simple_rnn_6/transpose:y:09sequential_21/simple_rnn_6/strided_slice_2/stack:output:0;sequential_21/simple_rnn_6/strided_slice_2/stack_1:output:0;sequential_21/simple_rnn_6/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџр*
shrink_axis_maskЯ
Bsequential_21/simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOpReadVariableOpKsequential_21_simple_rnn_6_simple_rnn_cell_6_matmul_readvariableop_resource*
_output_shapes
:	р@*
dtype0№
3sequential_21/simple_rnn_6/simple_rnn_cell_6/MatMulMatMul3sequential_21/simple_rnn_6/strided_slice_2:output:0Jsequential_21/simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@Ь
Csequential_21/simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOpReadVariableOpLsequential_21_simple_rnn_6_simple_rnn_cell_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0§
4sequential_21/simple_rnn_6/simple_rnn_cell_6/BiasAddBiasAdd=sequential_21/simple_rnn_6/simple_rnn_cell_6/MatMul:product:0Ksequential_21/simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@в
Dsequential_21/simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOpReadVariableOpMsequential_21_simple_rnn_6_simple_rnn_cell_6_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0ъ
5sequential_21/simple_rnn_6/simple_rnn_cell_6/MatMul_1MatMul)sequential_21/simple_rnn_6/zeros:output:0Lsequential_21/simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@ы
0sequential_21/simple_rnn_6/simple_rnn_cell_6/addAddV2=sequential_21/simple_rnn_6/simple_rnn_cell_6/BiasAdd:output:0?sequential_21/simple_rnn_6/simple_rnn_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@Ё
1sequential_21/simple_rnn_6/simple_rnn_cell_6/TanhTanh4sequential_21/simple_rnn_6/simple_rnn_cell_6/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@
8sequential_21/simple_rnn_6/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   
*sequential_21/simple_rnn_6/TensorArrayV2_1TensorListReserveAsequential_21/simple_rnn_6/TensorArrayV2_1/element_shape:output:03sequential_21/simple_rnn_6/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвa
sequential_21/simple_rnn_6/timeConst*
_output_shapes
: *
dtype0*
value	B : ~
3sequential_21/simple_rnn_6/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџo
-sequential_21/simple_rnn_6/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Й
 sequential_21/simple_rnn_6/whileWhile6sequential_21/simple_rnn_6/while/loop_counter:output:0<sequential_21/simple_rnn_6/while/maximum_iterations:output:0(sequential_21/simple_rnn_6/time:output:03sequential_21/simple_rnn_6/TensorArrayV2_1:handle:0)sequential_21/simple_rnn_6/zeros:output:03sequential_21/simple_rnn_6/strided_slice_1:output:0Rsequential_21/simple_rnn_6/TensorArrayUnstack/TensorListFromTensor:output_handle:0Ksequential_21_simple_rnn_6_simple_rnn_cell_6_matmul_readvariableop_resourceLsequential_21_simple_rnn_6_simple_rnn_cell_6_biasadd_readvariableop_resourceMsequential_21_simple_rnn_6_simple_rnn_cell_6_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *9
body1R/
-sequential_21_simple_rnn_6_while_body_1927914*9
cond1R/
-sequential_21_simple_rnn_6_while_cond_1927913*8
output_shapes'
%: : : : :џџџџџџџџџ@: : : : : *
parallel_iterations 
Ksequential_21/simple_rnn_6/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   
=sequential_21/simple_rnn_6/TensorArrayV2Stack/TensorListStackTensorListStack)sequential_21/simple_rnn_6/while:output:3Tsequential_21/simple_rnn_6/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ@*
element_dtype0
0sequential_21/simple_rnn_6/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ|
2sequential_21/simple_rnn_6/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: |
2sequential_21/simple_rnn_6/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
*sequential_21/simple_rnn_6/strided_slice_3StridedSliceFsequential_21/simple_rnn_6/TensorArrayV2Stack/TensorListStack:tensor:09sequential_21/simple_rnn_6/strided_slice_3/stack:output:0;sequential_21/simple_rnn_6/strided_slice_3/stack_1:output:0;sequential_21/simple_rnn_6/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*
shrink_axis_mask
+sequential_21/simple_rnn_6/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ч
&sequential_21/simple_rnn_6/transpose_1	TransposeFsequential_21/simple_rnn_6/TensorArrayV2Stack/TensorListStack:tensor:04sequential_21/simple_rnn_6/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@s
sequential_21/gru_9/ShapeShape*sequential_21/simple_rnn_6/transpose_1:y:0*
T0*
_output_shapes
:q
'sequential_21/gru_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)sequential_21/gru_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)sequential_21/gru_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
!sequential_21/gru_9/strided_sliceStridedSlice"sequential_21/gru_9/Shape:output:00sequential_21/gru_9/strided_slice/stack:output:02sequential_21/gru_9/strided_slice/stack_1:output:02sequential_21/gru_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"sequential_21/gru_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : Џ
 sequential_21/gru_9/zeros/packedPack*sequential_21/gru_9/strided_slice:output:0+sequential_21/gru_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:d
sequential_21/gru_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ј
sequential_21/gru_9/zerosFill)sequential_21/gru_9/zeros/packed:output:0(sequential_21/gru_9/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ 
'sequential_21/gru_9/Read/ReadVariableOpReadVariableOp0sequential_21_gru_9_read_readvariableop_resource*
_output_shapes

:@`*
dtype0
sequential_21/gru_9/IdentityIdentity/sequential_21/gru_9/Read/ReadVariableOp:value:0*
T0*
_output_shapes

:@`
)sequential_21/gru_9/Read_1/ReadVariableOpReadVariableOp2sequential_21_gru_9_read_1_readvariableop_resource*
_output_shapes

: `*
dtype0
sequential_21/gru_9/Identity_1Identity1sequential_21/gru_9/Read_1/ReadVariableOp:value:0*
T0*
_output_shapes

: `
)sequential_21/gru_9/Read_2/ReadVariableOpReadVariableOp2sequential_21_gru_9_read_2_readvariableop_resource*
_output_shapes

:`*
dtype0
sequential_21/gru_9/Identity_2Identity1sequential_21/gru_9/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes

:`
#sequential_21/gru_9/PartitionedCallPartitionedCall*sequential_21/simple_rnn_6/transpose_1:y:0"sequential_21/gru_9/zeros:output:0%sequential_21/gru_9/Identity:output:0'sequential_21/gru_9/Identity_1:output:0'sequential_21/gru_9/Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *S
_output_shapesA
?:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference_standard_gru_1928139Ђ
,sequential_21/dense_28/MatMul/ReadVariableOpReadVariableOp5sequential_21_dense_28_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Н
sequential_21/dense_28/MatMulMatMul,sequential_21/gru_9/PartitionedCall:output:04sequential_21/dense_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
-sequential_21/dense_28/BiasAdd/ReadVariableOpReadVariableOp6sequential_21_dense_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Л
sequential_21/dense_28/BiasAddBiasAdd'sequential_21/dense_28/MatMul:product:05sequential_21/dense_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЂ
,sequential_21/dense_29/MatMul/ReadVariableOpReadVariableOp5sequential_21_dense_29_matmul_readvariableop_resource*
_output_shapes

:*
dtype0И
sequential_21/dense_29/MatMulMatMul'sequential_21/dense_28/BiasAdd:output:04sequential_21/dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
-sequential_21/dense_29/BiasAdd/ReadVariableOpReadVariableOp6sequential_21_dense_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Л
sequential_21/dense_29/BiasAddBiasAdd'sequential_21/dense_29/MatMul:product:05sequential_21/dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџv
IdentityIdentity'sequential_21/dense_29/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџч
NoOpNoOp.^sequential_21/conv1d_8/BiasAdd/ReadVariableOp:^sequential_21/conv1d_8/Conv1D/ExpandDims_1/ReadVariableOp.^sequential_21/dense_28/BiasAdd/ReadVariableOp-^sequential_21/dense_28/MatMul/ReadVariableOp.^sequential_21/dense_29/BiasAdd/ReadVariableOp-^sequential_21/dense_29/MatMul/ReadVariableOp(^sequential_21/gru_9/Read/ReadVariableOp*^sequential_21/gru_9/Read_1/ReadVariableOp*^sequential_21/gru_9/Read_2/ReadVariableOpD^sequential_21/simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOpC^sequential_21/simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOpE^sequential_21/simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOp!^sequential_21/simple_rnn_6/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ: : : : : : : : : : : : 2^
-sequential_21/conv1d_8/BiasAdd/ReadVariableOp-sequential_21/conv1d_8/BiasAdd/ReadVariableOp2v
9sequential_21/conv1d_8/Conv1D/ExpandDims_1/ReadVariableOp9sequential_21/conv1d_8/Conv1D/ExpandDims_1/ReadVariableOp2^
-sequential_21/dense_28/BiasAdd/ReadVariableOp-sequential_21/dense_28/BiasAdd/ReadVariableOp2\
,sequential_21/dense_28/MatMul/ReadVariableOp,sequential_21/dense_28/MatMul/ReadVariableOp2^
-sequential_21/dense_29/BiasAdd/ReadVariableOp-sequential_21/dense_29/BiasAdd/ReadVariableOp2\
,sequential_21/dense_29/MatMul/ReadVariableOp,sequential_21/dense_29/MatMul/ReadVariableOp2R
'sequential_21/gru_9/Read/ReadVariableOp'sequential_21/gru_9/Read/ReadVariableOp2V
)sequential_21/gru_9/Read_1/ReadVariableOp)sequential_21/gru_9/Read_1/ReadVariableOp2V
)sequential_21/gru_9/Read_2/ReadVariableOp)sequential_21/gru_9/Read_2/ReadVariableOp2
Csequential_21/simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOpCsequential_21/simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOp2
Bsequential_21/simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOpBsequential_21/simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOp2
Dsequential_21/simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOpDsequential_21/simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOp2D
 sequential_21/simple_rnn_6/while sequential_21/simple_rnn_6/while:[ W
+
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameconv1d_8_input
Џ,
Ы
while_body_1932171
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
8while_simple_rnn_cell_6_matmul_readvariableop_resource_0:	р@G
9while_simple_rnn_cell_6_biasadd_readvariableop_resource_0:@L
:while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
6while_simple_rnn_cell_6_matmul_readvariableop_resource:	р@E
7while_simple_rnn_cell_6_biasadd_readvariableop_resource:@J
8while_simple_rnn_cell_6_matmul_1_readvariableop_resource:@@Ђ.while/simple_rnn_cell_6/BiasAdd/ReadVariableOpЂ-while/simple_rnn_cell_6/MatMul/ReadVariableOpЂ/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџр  Ї
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџр*
element_dtype0Ї
-while/simple_rnn_cell_6/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_6_matmul_readvariableop_resource_0*
_output_shapes
:	р@*
dtype0У
while/simple_rnn_cell_6/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@Є
.while/simple_rnn_cell_6/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_6_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0О
while/simple_rnn_cell_6/BiasAddBiasAdd(while/simple_rnn_cell_6/MatMul:product:06while/simple_rnn_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@Њ
/while/simple_rnn_cell_6/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0Њ
 while/simple_rnn_cell_6/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@Ќ
while/simple_rnn_cell_6/addAddV2(while/simple_rnn_cell_6/BiasAdd:output:0*while/simple_rnn_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@w
while/simple_rnn_cell_6/TanhTanhwhile/simple_rnn_cell_6/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Щ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_6/Tanh:y:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: }
while/Identity_4Identity while/simple_rnn_cell_6/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@п

while/NoOpNoOp/^while/simple_rnn_cell_6/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_6/MatMul/ReadVariableOp0^while/simple_rnn_cell_6/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_6_biasadd_readvariableop_resource9while_simple_rnn_cell_6_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_6_matmul_1_readvariableop_resource:while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_6_matmul_readvariableop_resource8while_simple_rnn_cell_6_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ@: : : : : 2`
.while/simple_rnn_cell_6/BiasAdd/ReadVariableOp.while/simple_rnn_cell_6/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_6/MatMul/ReadVariableOp-while/simple_rnn_cell_6/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ@:

_output_shapes
: :

_output_shapes
: 

И
/__inference_sequential_21_layer_call_fn_1930875

inputs
unknown: 
	unknown_0: 
	unknown_1:	р@
	unknown_2:@
	unknown_3:@@
	unknown_4:@`
	unknown_5: `
	unknown_6:`
	unknown_7: 
	unknown_8:
	unknown_9:

unknown_10:
identityЂStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_21_layer_call_and_return_conditional_losses_1930024o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ў
к

<__inference___backward_gpu_gru_with_fallback_1928216_1928352
placeholder
placeholder_1
placeholder_2
placeholder_33
/gradients_expanddims_1_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:џџџџџџџџџ d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:џџџџџџџџџ `
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:џџџџџџџџџ O
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 
!gradients/ExpandDims_1_grad/ShapeShape/gradients_expanddims_1_grad_shape_strided_slice*
T0*
_output_shapes
:Њ
#gradients/ExpandDims_1_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:Є
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Х
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_1_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:џџџџџџџџџ }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџ *
shrink_axis_maska
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*K
_output_shapes9
7:џџџџџџџџџ@:џџџџџџџџџ : :РI*
rnn_modegru
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:а
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:Х
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ \
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:g
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: 
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::ц
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:щ
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
: щ
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
: щ
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: щ
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: ь
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: ь
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   Ѓ
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   Ѕ
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   Ѕ
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ѕ
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ѕ
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ѕ
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  h
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ё
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ё
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ё
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ѓ
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Є
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Є
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: 
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:З
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ 
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:З
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ 
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:З
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ 
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:З
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:З
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:З
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Р
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:@`
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

: `m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   `   Ё
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:`r
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ e

Identity_2Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes

:@`g

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes

: `h

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes

:`"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ : :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ : ::џџџџџџџџџ@:џџџџџџџџџ : :РI::џџџџџџџџџ : ::::::: : : *<
api_implements*(gru_73e98d14-b186-4092-be4c-44eb47643dc0*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_gru_with_fallback_1928351*
go_backwards( *

time_major( :- )
'
_output_shapes
:џџџџџџџџџ :1-
+
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :1-
+
_output_shapes
:џџџџџџџџџ :1-
+
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
::1	-
+
_output_shapes
:џџџџџџџџџ@:1
-
+
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :!

_output_shapes	
:РI: 

_output_shapes
::-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
П
з

J__inference_sequential_21_layer_call_and_return_conditional_losses_1931928

inputsJ
4conv1d_8_conv1d_expanddims_1_readvariableop_resource: 6
(conv1d_8_biasadd_readvariableop_resource: P
=simple_rnn_6_simple_rnn_cell_6_matmul_readvariableop_resource:	р@L
>simple_rnn_6_simple_rnn_cell_6_biasadd_readvariableop_resource:@Q
?simple_rnn_6_simple_rnn_cell_6_matmul_1_readvariableop_resource:@@4
"gru_9_read_readvariableop_resource:@`6
$gru_9_read_1_readvariableop_resource: `6
$gru_9_read_2_readvariableop_resource:`9
'dense_28_matmul_readvariableop_resource: 6
(dense_28_biasadd_readvariableop_resource:9
'dense_29_matmul_readvariableop_resource:6
(dense_29_biasadd_readvariableop_resource:
identityЂconv1d_8/BiasAdd/ReadVariableOpЂ+conv1d_8/Conv1D/ExpandDims_1/ReadVariableOpЂdense_28/BiasAdd/ReadVariableOpЂdense_28/MatMul/ReadVariableOpЂdense_29/BiasAdd/ReadVariableOpЂdense_29/MatMul/ReadVariableOpЂgru_9/Read/ReadVariableOpЂgru_9/Read_1/ReadVariableOpЂgru_9/Read_2/ReadVariableOpЂ5simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOpЂ4simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOpЂ6simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOpЂsimple_rnn_6/whilei
conv1d_8/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ
conv1d_8/Conv1D/ExpandDims
ExpandDimsinputs'conv1d_8/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџЄ
+conv1d_8/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_8_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0b
 conv1d_8/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Л
conv1d_8/Conv1D/ExpandDims_1
ExpandDims3conv1d_8/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_8/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: Ш
conv1d_8/Conv1DConv2D#conv1d_8/Conv1D/ExpandDims:output:0%conv1d_8/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides

conv1d_8/Conv1D/SqueezeSqueezeconv1d_8/Conv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџ *
squeeze_dims

§џџџџџџџџ
conv1d_8/BiasAdd/ReadVariableOpReadVariableOp(conv1d_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv1d_8/BiasAddBiasAdd conv1d_8/Conv1D/Squeeze:output:0'conv1d_8/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ f
conv1d_8/ReluReluconv1d_8/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ `
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџр  
flatten_9/ReshapeReshapeconv1d_8/Relu:activations:0flatten_9/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџр`
repeat_vector_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Є
repeat_vector_2/ExpandDims
ExpandDimsflatten_9/Reshape:output:0'repeat_vector_2/ExpandDims/dim:output:0*
T0*,
_output_shapes
:џџџџџџџџџрj
repeat_vector_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"         
repeat_vector_2/TileTile#repeat_vector_2/ExpandDims:output:0repeat_vector_2/stack:output:0*
T0*,
_output_shapes
:џџџџџџџџџр_
simple_rnn_6/ShapeShaperepeat_vector_2/Tile:output:0*
T0*
_output_shapes
:j
 simple_rnn_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"simple_rnn_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"simple_rnn_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
simple_rnn_6/strided_sliceStridedSlicesimple_rnn_6/Shape:output:0)simple_rnn_6/strided_slice/stack:output:0+simple_rnn_6/strided_slice/stack_1:output:0+simple_rnn_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
simple_rnn_6/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@
simple_rnn_6/zeros/packedPack#simple_rnn_6/strided_slice:output:0$simple_rnn_6/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
simple_rnn_6/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
simple_rnn_6/zerosFill"simple_rnn_6/zeros/packed:output:0!simple_rnn_6/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@p
simple_rnn_6/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
simple_rnn_6/transpose	Transposerepeat_vector_2/Tile:output:0$simple_rnn_6/transpose/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџр^
simple_rnn_6/Shape_1Shapesimple_rnn_6/transpose:y:0*
T0*
_output_shapes
:l
"simple_rnn_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
simple_rnn_6/strided_slice_1StridedSlicesimple_rnn_6/Shape_1:output:0+simple_rnn_6/strided_slice_1/stack:output:0-simple_rnn_6/strided_slice_1/stack_1:output:0-simple_rnn_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(simple_rnn_6/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџл
simple_rnn_6/TensorArrayV2TensorListReserve1simple_rnn_6/TensorArrayV2/element_shape:output:0%simple_rnn_6/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
Bsimple_rnn_6/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџр  
4simple_rnn_6/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_6/transpose:y:0Ksimple_rnn_6/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвl
"simple_rnn_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ћ
simple_rnn_6/strided_slice_2StridedSlicesimple_rnn_6/transpose:y:0+simple_rnn_6/strided_slice_2/stack:output:0-simple_rnn_6/strided_slice_2/stack_1:output:0-simple_rnn_6/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџр*
shrink_axis_maskГ
4simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOpReadVariableOp=simple_rnn_6_simple_rnn_cell_6_matmul_readvariableop_resource*
_output_shapes
:	р@*
dtype0Ц
%simple_rnn_6/simple_rnn_cell_6/MatMulMatMul%simple_rnn_6/strided_slice_2:output:0<simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@А
5simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_6_simple_rnn_cell_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0г
&simple_rnn_6/simple_rnn_cell_6/BiasAddBiasAdd/simple_rnn_6/simple_rnn_cell_6/MatMul:product:0=simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@Ж
6simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_6_simple_rnn_cell_6_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Р
'simple_rnn_6/simple_rnn_cell_6/MatMul_1MatMulsimple_rnn_6/zeros:output:0>simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@С
"simple_rnn_6/simple_rnn_cell_6/addAddV2/simple_rnn_6/simple_rnn_cell_6/BiasAdd:output:01simple_rnn_6/simple_rnn_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@
#simple_rnn_6/simple_rnn_cell_6/TanhTanh&simple_rnn_6/simple_rnn_cell_6/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@{
*simple_rnn_6/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   п
simple_rnn_6/TensorArrayV2_1TensorListReserve3simple_rnn_6/TensorArrayV2_1/element_shape:output:0%simple_rnn_6/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвS
simple_rnn_6/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%simple_rnn_6/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџa
simple_rnn_6/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
simple_rnn_6/whileWhile(simple_rnn_6/while/loop_counter:output:0.simple_rnn_6/while/maximum_iterations:output:0simple_rnn_6/time:output:0%simple_rnn_6/TensorArrayV2_1:handle:0simple_rnn_6/zeros:output:0%simple_rnn_6/strided_slice_1:output:0Dsimple_rnn_6/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_6_simple_rnn_cell_6_matmul_readvariableop_resource>simple_rnn_6_simple_rnn_cell_6_biasadd_readvariableop_resource?simple_rnn_6_simple_rnn_cell_6_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *+
body#R!
simple_rnn_6_while_body_1931476*+
cond#R!
simple_rnn_6_while_cond_1931475*8
output_shapes'
%: : : : :џџџџџџџџџ@: : : : : *
parallel_iterations 
=simple_rnn_6/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   щ
/simple_rnn_6/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_6/while:output:3Fsimple_rnn_6/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ@*
element_dtype0u
"simple_rnn_6/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџn
$simple_rnn_6/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_6/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ш
simple_rnn_6/strided_slice_3StridedSlice8simple_rnn_6/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_6/strided_slice_3/stack:output:0-simple_rnn_6/strided_slice_3/stack_1:output:0-simple_rnn_6/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*
shrink_axis_maskr
simple_rnn_6/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Н
simple_rnn_6/transpose_1	Transpose8simple_rnn_6/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_6/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@W
gru_9/ShapeShapesimple_rnn_6/transpose_1:y:0*
T0*
_output_shapes
:c
gru_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:я
gru_9/strided_sliceStridedSlicegru_9/Shape:output:0"gru_9/strided_slice/stack:output:0$gru_9/strided_slice/stack_1:output:0$gru_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
gru_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 
gru_9/zeros/packedPackgru_9/strided_slice:output:0gru_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:V
gru_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
gru_9/zerosFillgru_9/zeros/packed:output:0gru_9/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ |
gru_9/Read/ReadVariableOpReadVariableOp"gru_9_read_readvariableop_resource*
_output_shapes

:@`*
dtype0f
gru_9/IdentityIdentity!gru_9/Read/ReadVariableOp:value:0*
T0*
_output_shapes

:@`
gru_9/Read_1/ReadVariableOpReadVariableOp$gru_9_read_1_readvariableop_resource*
_output_shapes

: `*
dtype0j
gru_9/Identity_1Identity#gru_9/Read_1/ReadVariableOp:value:0*
T0*
_output_shapes

: `
gru_9/Read_2/ReadVariableOpReadVariableOp$gru_9_read_2_readvariableop_resource*
_output_shapes

:`*
dtype0j
gru_9/Identity_2Identity#gru_9/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes

:`П
gru_9/PartitionedCallPartitionedCallsimple_rnn_6/transpose_1:y:0gru_9/zeros:output:0gru_9/Identity:output:0gru_9/Identity_1:output:0gru_9/Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *S
_output_shapesA
?:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference_standard_gru_1931701
dense_28/MatMul/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_28/MatMulMatMulgru_9/PartitionedCall:output:0&dense_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_28/BiasAddBiasAdddense_28/MatMul:product:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_29/MatMul/ReadVariableOpReadVariableOp'dense_29_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_29/MatMulMatMuldense_28/BiasAdd:output:0&dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџh
IdentityIdentitydense_29/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџБ
NoOpNoOp ^conv1d_8/BiasAdd/ReadVariableOp,^conv1d_8/Conv1D/ExpandDims_1/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOp^gru_9/Read/ReadVariableOp^gru_9/Read_1/ReadVariableOp^gru_9/Read_2/ReadVariableOp6^simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOp5^simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOp7^simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOp^simple_rnn_6/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ: : : : : : : : : : : : 2B
conv1d_8/BiasAdd/ReadVariableOpconv1d_8/BiasAdd/ReadVariableOp2Z
+conv1d_8/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_8/Conv1D/ExpandDims_1/ReadVariableOp2B
dense_28/BiasAdd/ReadVariableOpdense_28/BiasAdd/ReadVariableOp2@
dense_28/MatMul/ReadVariableOpdense_28/MatMul/ReadVariableOp2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2@
dense_29/MatMul/ReadVariableOpdense_29/MatMul/ReadVariableOp26
gru_9/Read/ReadVariableOpgru_9/Read/ReadVariableOp2:
gru_9/Read_1/ReadVariableOpgru_9/Read_1/ReadVariableOp2:
gru_9/Read_2/ReadVariableOpgru_9/Read_2/ReadVariableOp2n
5simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOp5simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOp2l
4simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOp4simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOp2p
6simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOp6simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOp2(
simple_rnn_6/whilesimple_rnn_6/while:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Џ,
Ы
while_body_1932063
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
8while_simple_rnn_cell_6_matmul_readvariableop_resource_0:	р@G
9while_simple_rnn_cell_6_biasadd_readvariableop_resource_0:@L
:while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
6while_simple_rnn_cell_6_matmul_readvariableop_resource:	р@E
7while_simple_rnn_cell_6_biasadd_readvariableop_resource:@J
8while_simple_rnn_cell_6_matmul_1_readvariableop_resource:@@Ђ.while/simple_rnn_cell_6/BiasAdd/ReadVariableOpЂ-while/simple_rnn_cell_6/MatMul/ReadVariableOpЂ/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџр  Ї
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџр*
element_dtype0Ї
-while/simple_rnn_cell_6/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_6_matmul_readvariableop_resource_0*
_output_shapes
:	р@*
dtype0У
while/simple_rnn_cell_6/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@Є
.while/simple_rnn_cell_6/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_6_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0О
while/simple_rnn_cell_6/BiasAddBiasAdd(while/simple_rnn_cell_6/MatMul:product:06while/simple_rnn_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@Њ
/while/simple_rnn_cell_6/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0Њ
 while/simple_rnn_cell_6/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@Ќ
while/simple_rnn_cell_6/addAddV2(while/simple_rnn_cell_6/BiasAdd:output:0*while/simple_rnn_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@w
while/simple_rnn_cell_6/TanhTanhwhile/simple_rnn_cell_6/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Щ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_6/Tanh:y:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: }
while/Identity_4Identity while/simple_rnn_cell_6/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@п

while/NoOpNoOp/^while/simple_rnn_cell_6/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_6/MatMul/ReadVariableOp0^while/simple_rnn_cell_6/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_6_biasadd_readvariableop_resource9while_simple_rnn_cell_6_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_6_matmul_1_readvariableop_resource:while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_6_matmul_readvariableop_resource8while_simple_rnn_cell_6_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ@: : : : : 2`
.while/simple_rnn_cell_6/BiasAdd/ReadVariableOp.while/simple_rnn_cell_6/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_6/MatMul/ReadVariableOp-while/simple_rnn_cell_6/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ@:

_output_shapes
: :

_output_shapes
: 
ў
к

<__inference___backward_gpu_gru_with_fallback_1933871_1934007
placeholder
placeholder_1
placeholder_2
placeholder_33
/gradients_expanddims_1_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:џџџџџџџџџ d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:џџџџџџџџџ `
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:џџџџџџџџџ O
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 
!gradients/ExpandDims_1_grad/ShapeShape/gradients_expanddims_1_grad_shape_strided_slice*
T0*
_output_shapes
:Њ
#gradients/ExpandDims_1_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:Є
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Х
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_1_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:џџџџџџџџџ }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџ *
shrink_axis_maska
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*K
_output_shapes9
7:џџџџџџџџџ@:џџџџџџџџџ : :РI*
rnn_modegru
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:а
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:Х
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ \
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:g
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: 
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::ц
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:щ
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
: щ
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
: щ
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: щ
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: ь
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: ь
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   Ѓ
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   Ѕ
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   Ѕ
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ѕ
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ѕ
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ѕ
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  h
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ё
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ё
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ё
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ѓ
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Є
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Є
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: 
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:З
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ 
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:З
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ 
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:З
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ 
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:З
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:З
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:З
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Р
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:@`
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

: `m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   `   Ё
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:`r
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ e

Identity_2Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes

:@`g

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes

: `h

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes

:`"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ : :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ : ::џџџџџџџџџ@:џџџџџџџџџ : :РI::џџџџџџџџџ : ::::::: : : *<
api_implements*(gru_b015bfb0-6782-4f52-830c-ea8adfe5cd9c*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_gru_with_fallback_1934006*
go_backwards( *

time_major( :- )
'
_output_shapes
:џџџџџџџџџ :1-
+
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :1-
+
_output_shapes
:џџџџџџџџџ :1-
+
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
::1	-
+
_output_shapes
:џџџџџџџџџ@:1
-
+
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :!

_output_shapes	
:РI: 

_output_shapes
::-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
>
Є
 __inference_standard_gru_1930247

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3M
unstackUnpackbias*
T0* 
_output_shapes
:`:`*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*
shrink_axis_maskd
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:џџџџџџџџџ`h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ѓ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split^
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:џџџџџџџџџ`l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Љ
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ S
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:џџџџџџџџџ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ Q
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ў
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :џџџџџџџџџ : : :@`:`: `:`* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_1930157*
condR
while_cond_1930156*R
output_shapesA
?: : : : :џџџџџџџџџ : : :@`:`: `:`*
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ ]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:џџџџџџџџџ I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:џџџџџџџџџ@:џџџџџџџџџ :@`: `:`*<
api_implements*(gru_f821b5a3-099c-46c6-b85a-23e6bbf0193a*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
Ц
к

<__inference___backward_gpu_gru_with_fallback_1929306_1929442
placeholder
placeholder_1
placeholder_2
placeholder_33
/gradients_expanddims_1_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:џџџџџџџџџ d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:џџџџџџџџџ `
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:џџџџџџџџџ O
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 
!gradients/ExpandDims_1_grad/ShapeShape/gradients_expanddims_1_grad_shape_strided_slice*
T0*
_output_shapes
:Њ
#gradients/ExpandDims_1_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:Є
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Х
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_1_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:џџџџџџџџџ }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:Ђ
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ *
shrink_axis_maska
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*T
_output_shapesB
@:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџ : :РI*
rnn_modegru
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:й
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:Х
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ \
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:g
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: 
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::ц
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:щ
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
: щ
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
: щ
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: щ
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: ь
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: ь
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   Ѓ
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   Ѕ
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   Ѕ
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ѕ
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ѕ
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ѕ
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  h
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ё
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ё
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ё
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ѓ
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Є
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Є
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: 
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:З
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ 
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:З
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ 
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:З
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ 
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:З
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:З
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:З
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Р
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:@`
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

: `m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   `   Ё
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:`{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ e

Identity_2Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes

:@`g

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes

: `h

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes

:`"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*Љ
_input_shapes
:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ : :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџџџџџџџџџџ : ::џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџ : :РI::џџџџџџџџџ : ::::::: : : *<
api_implements*(gru_7ffb1024-6f26-45b8-a31e-290afcdaacd2*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_gru_with_fallback_1929441*
go_backwards( *

time_major( :- )
'
_output_shapes
:џџџџџџџџџ :1-
+
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :1-
+
_output_shapes
:џџџџџџџџџ ::6
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ :

_output_shapes
: :

_output_shapes
:::	6
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@:1
-
+
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :!

_output_shapes	
:РI: 

_output_shapes
::-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
>
К
'__forward_gpu_gru_with_fallback_1928351

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dimc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџV
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:РS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Џ
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    Ю
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*H
_output_shapes6
4:џџџџџџџџџ :џџџџџџџџџ : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:О
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџ *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_1
ExpandDimsstrided_slice_0:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:џџџџџџџџџ c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџ I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"

cudnnrnn_0CudnnRNN:output:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:џџџџџџџџџ@:џџџџџџџџџ :@`: `:`*<
api_implements*(gru_73e98d14-b186-4092-be4c-44eb47643dc0*
api_preferred_deviceGPU*X
backward_function_name><__inference___backward_gpu_gru_with_fallback_1928216_1928352*
go_backwards( *

time_major( :S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
Ћ=
Н
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_1932453

inputsC
0simple_rnn_cell_6_matmul_readvariableop_resource:	р@?
1simple_rnn_cell_6_biasadd_readvariableop_resource:@D
2simple_rnn_cell_6_matmul_1_readvariableop_resource:@@
identityЂ(simple_rnn_cell_6/BiasAdd/ReadVariableOpЂ'simple_rnn_cell_6/MatMul/ReadVariableOpЂ)simple_rnn_cell_6/MatMul_1/ReadVariableOpЂwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџрD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџр  р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџр*
shrink_axis_mask
'simple_rnn_cell_6/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_6_matmul_readvariableop_resource*
_output_shapes
:	р@*
dtype0
simple_rnn_cell_6/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
(simple_rnn_cell_6/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ќ
simple_rnn_cell_6/BiasAddBiasAdd"simple_rnn_cell_6/MatMul:product:00simple_rnn_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
)simple_rnn_cell_6/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_6_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_6/MatMul_1MatMulzeros:output:01simple_rnn_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
simple_rnn_cell_6/addAddV2"simple_rnn_cell_6/BiasAdd:output:0$simple_rnn_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@k
simple_rnn_cell_6/TanhTanhsimple_rnn_cell_6/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : к
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_6_matmul_readvariableop_resource1simple_rnn_cell_6_biasadd_readvariableop_resource2simple_rnn_cell_6_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1932387*
condR
while_cond_1932386*8
output_shapes'
%: : : : :џџџџџџџџџ@: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   Т
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ@Я
NoOpNoOp)^simple_rnn_cell_6/BiasAdd/ReadVariableOp(^simple_rnn_cell_6/MatMul/ReadVariableOp*^simple_rnn_cell_6/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџр: : : 2T
(simple_rnn_cell_6/BiasAdd/ReadVariableOp(simple_rnn_cell_6/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_6/MatMul/ReadVariableOp'simple_rnn_cell_6/MatMul/ReadVariableOp2V
)simple_rnn_cell_6/MatMul_1/ReadVariableOp)simple_rnn_cell_6/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:џџџџџџџџџр
 
_user_specified_nameinputs
Ё
Р
/__inference_sequential_21_layer_call_fn_1930739
conv1d_8_input
unknown: 
	unknown_0: 
	unknown_1:	р@
	unknown_2:@
	unknown_3:@@
	unknown_4:@`
	unknown_5: `
	unknown_6:`
	unknown_7: 
	unknown_8:
	unknown_9:

unknown_10:
identityЂStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallconv1d_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_21_layer_call_and_return_conditional_losses_1930683o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameconv1d_8_input
Ш	
і
E__inference_dense_28_layer_call_and_return_conditional_losses_1934028

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
і
Е
'__inference_gru_9_layer_call_fn_1932486

inputs
unknown:@`
	unknown_0: `
	unknown_1:`
identityЂStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_gru_9_layer_call_and_return_conditional_losses_1929983o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
>
К
'__forward_gpu_gru_with_fallback_1931401

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dimc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџV
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:РS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Џ
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    Ю
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*H
_output_shapes6
4:џџџџџџџџџ :џџџџџџџџџ : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:О
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџ *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_1
ExpandDimsstrided_slice_0:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:џџџџџџџџџ c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџ I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"

cudnnrnn_0CudnnRNN:output:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:џџџџџџџџџ@:џџџџџџџџџ :@`: `:`*<
api_implements*(gru_7cdd8818-ca5e-4d7b-98a6-7015d31c0e59*
api_preferred_deviceGPU*X
backward_function_name><__inference___backward_gpu_gru_with_fallback_1931266_1931402*
go_backwards( *

time_major( :S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
э"
Щ
J__inference_sequential_21_layer_call_and_return_conditional_losses_1930774
conv1d_8_input&
conv1d_8_1930742: 
conv1d_8_1930744: '
simple_rnn_6_1930749:	р@"
simple_rnn_6_1930751:@&
simple_rnn_6_1930753:@@
gru_9_1930756:@`
gru_9_1930758: `
gru_9_1930760:`"
dense_28_1930763: 
dense_28_1930765:"
dense_29_1930768:
dense_29_1930770:
identityЂ conv1d_8/StatefulPartitionedCallЂ dense_28/StatefulPartitionedCallЂ dense_29/StatefulPartitionedCallЂgru_9/StatefulPartitionedCallЂ$simple_rnn_6/StatefulPartitionedCallџ
 conv1d_8/StatefulPartitionedCallStatefulPartitionedCallconv1d_8_inputconv1d_8_1930742conv1d_8_1930744*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv1d_8_layer_call_and_return_conditional_losses_1929476п
flatten_9/PartitionedCallPartitionedCall)conv1d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџр* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_9_layer_call_and_return_conditional_losses_1929488ш
repeat_vector_2/PartitionedCallPartitionedCall"flatten_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџр* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_repeat_vector_2_layer_call_and_return_conditional_losses_1928378С
$simple_rnn_6/StatefulPartitionedCallStatefulPartitionedCall(repeat_vector_2/PartitionedCall:output:0simple_rnn_6_1930749simple_rnn_6_1930751simple_rnn_6_1930753*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_1929598
gru_9/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_6/StatefulPartitionedCall:output:0gru_9_1930756gru_9_1930758gru_9_1930760*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_gru_9_layer_call_and_return_conditional_losses_1929983
 dense_28/StatefulPartitionedCallStatefulPartitionedCall&gru_9/StatefulPartitionedCall:output:0dense_28_1930763dense_28_1930765*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_28_layer_call_and_return_conditional_losses_1930001
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0dense_29_1930768dense_29_1930770*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_29_layer_call_and_return_conditional_losses_1930017x
IdentityIdentity)dense_29/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџі
NoOpNoOp!^conv1d_8/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall^gru_9/StatefulPartitionedCall%^simple_rnn_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ: : : : : : : : : : : : 2D
 conv1d_8/StatefulPartitionedCall conv1d_8/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2>
gru_9/StatefulPartitionedCallgru_9/StatefulPartitionedCall2L
$simple_rnn_6/StatefulPartitionedCall$simple_rnn_6/StatefulPartitionedCall:[ W
+
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameconv1d_8_input
н-
у
while_body_1932570
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ@*
element_dtype0
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ`{
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:џџџџџџџџџ`W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Е
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ`
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:џџџџџџџџџ`Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Л
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_splitr
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ Y
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ t
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ ]
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ o
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ k
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ U

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ l
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ P
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?k
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ c
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ h
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/add_3:z:0*
_output_shapes
: *
element_dtype0:щшвO
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ "4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?: : : : :џџџџџџџџџ : : :@`:`: `:`: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@`: 

_output_shapes
:`:$	 

_output_shapes

: `: 


_output_shapes
:`
>
Є
 __inference_standard_gru_1933416

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3M
unstackUnpackbias*
T0* 
_output_shapes
:`:`*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*
shrink_axis_maskd
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:џџџџџџџџџ`h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ѓ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split^
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:џџџџџџџџџ`l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Љ
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ S
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:џџџџџџџџџ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ Q
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ў
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :џџџџџџџџџ : : :@`:`: `:`* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_1933326*
condR
while_cond_1933325*R
output_shapesA
?: : : : :џџџџџџџџџ : : :@`:`: `:`*
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ ]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:џџџџџџџџџ I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:џџџџџџџџџ@:џџџџџџџџџ :@`: `:`*<
api_implements*(gru_351bbcc5-99cd-49ef-abe4-0838c20dc796*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
н-
у
while_body_1932948
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ@*
element_dtype0
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ`{
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:џџџџџџџџџ`W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Е
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ`
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:џџџџџџџџџ`Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Л
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_splitr
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ Y
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ t
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ ]
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ o
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ k
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ U

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ l
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ P
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?k
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ c
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ h
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/add_3:z:0*
_output_shapes
: *
element_dtype0:щшвO
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ "4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?: : : : :џџџџџџџџџ : : :@`:`: `:`: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@`: 

_output_shapes
:`:$	 

_output_shapes

: `: 


_output_shapes
:`
З>
К
'__forward_gpu_gru_with_fallback_1929441

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dimc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџV
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:РS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Џ
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    з
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*Q
_output_shapes?
=:џџџџџџџџџџџџџџџџџџ :џџџџџџџџџ : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:О
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџ *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_1
ExpandDimsstrided_slice_0:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:џџџџџџџџџ c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџ I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"

cudnnrnn_0CudnnRNN:output:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџ :@`: `:`*<
api_implements*(gru_7ffb1024-6f26-45b8-a31e-290afcdaacd2*
api_preferred_deviceGPU*X
backward_function_name><__inference___backward_gpu_gru_with_fallback_1929306_1929442*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
О

м
3__inference_simple_rnn_cell_6_layer_call_fn_1934061

inputs
states_0
unknown:	р@
	unknown_0:@
	unknown_1:@@
identity

identity_1ЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ@:џџџџџџџџџ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_simple_rnn_cell_6_layer_call_and_return_conditional_losses_1928429o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:џџџџџџџџџр:џџџџџџџџџ@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџр
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ@
"
_user_specified_name
states/0
ѕ
П
B__inference_gru_9_layer_call_and_return_conditional_losses_1932875
inputs_0.
read_readvariableop_resource:@`0
read_1_readvariableop_resource: `0
read_2_readvariableop_resource:`

identity_3ЂRead/ReadVariableOpЂRead_1/ReadVariableOpЂRead_2/ReadVariableOp=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ p
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:@`*
dtype0Z
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:@`t
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

: `*
dtype0^

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

: `t
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes

:`*
dtype0^

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes

:`
PartitionedCallPartitionedCallinputs_0zeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *S
_output_shapesA
?:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference_standard_gru_1932660i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ@: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
"
_user_specified_name
inputs/0
Ћ
G
+__inference_flatten_9_layer_call_fn_1931958

inputs
identityВ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџр* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_9_layer_call_and_return_conditional_losses_1929488a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџр"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ :S O
+
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs

И
/__inference_sequential_21_layer_call_fn_1930904

inputs
unknown: 
	unknown_0: 
	unknown_1:	р@
	unknown_2:@
	unknown_3:@@
	unknown_4:@`
	unknown_5: `
	unknown_6:`
	unknown_7: 
	unknown_8:
	unknown_9:

unknown_10:
identityЂStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_21_layer_call_and_return_conditional_losses_1930683o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Џ,
Ы
while_body_1929532
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
8while_simple_rnn_cell_6_matmul_readvariableop_resource_0:	р@G
9while_simple_rnn_cell_6_biasadd_readvariableop_resource_0:@L
:while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
6while_simple_rnn_cell_6_matmul_readvariableop_resource:	р@E
7while_simple_rnn_cell_6_biasadd_readvariableop_resource:@J
8while_simple_rnn_cell_6_matmul_1_readvariableop_resource:@@Ђ.while/simple_rnn_cell_6/BiasAdd/ReadVariableOpЂ-while/simple_rnn_cell_6/MatMul/ReadVariableOpЂ/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџр  Ї
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџр*
element_dtype0Ї
-while/simple_rnn_cell_6/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_6_matmul_readvariableop_resource_0*
_output_shapes
:	р@*
dtype0У
while/simple_rnn_cell_6/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@Є
.while/simple_rnn_cell_6/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_6_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0О
while/simple_rnn_cell_6/BiasAddBiasAdd(while/simple_rnn_cell_6/MatMul:product:06while/simple_rnn_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@Њ
/while/simple_rnn_cell_6/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0Њ
 while/simple_rnn_cell_6/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@Ќ
while/simple_rnn_cell_6/addAddV2(while/simple_rnn_cell_6/BiasAdd:output:0*while/simple_rnn_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@w
while/simple_rnn_cell_6/TanhTanhwhile/simple_rnn_cell_6/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Щ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_6/Tanh:y:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: }
while/Identity_4Identity while/simple_rnn_cell_6/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@п

while/NoOpNoOp/^while/simple_rnn_cell_6/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_6/MatMul/ReadVariableOp0^while/simple_rnn_cell_6/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_6_biasadd_readvariableop_resource9while_simple_rnn_cell_6_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_6_matmul_1_readvariableop_resource:while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_6_matmul_readvariableop_resource8while_simple_rnn_cell_6_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ@: : : : : 2`
.while/simple_rnn_cell_6/BiasAdd/ReadVariableOp.while/simple_rnn_cell_6/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_6/MatMul/ReadVariableOp-while/simple_rnn_cell_6/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ@:

_output_shapes
: :

_output_shapes
: 
Є>
Є
 __inference_standard_gru_1928840

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3M
unstackUnpackbias*
T0* 
_output_shapes
:`:`*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*
shrink_axis_maskd
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:џџџџџџџџџ`h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ѓ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split^
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:џџџџџџџџџ`l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Љ
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ S
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:џџџџџџџџџ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ Q
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ў
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :џџџџџџџџџ : : :@`:`: `:`* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_1928750*
condR
while_cond_1928749*R
output_shapesA
?: : : : :џџџџџџџџџ : : :@`:`: `:`*
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ ]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:џџџџџџџџџ I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџ :@`: `:`*<
api_implements*(gru_e1cf211a-1d8c-47c6-ac5e-267262ea26df*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
Џ,
Ы
while_body_1930526
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
8while_simple_rnn_cell_6_matmul_readvariableop_resource_0:	р@G
9while_simple_rnn_cell_6_biasadd_readvariableop_resource_0:@L
:while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
6while_simple_rnn_cell_6_matmul_readvariableop_resource:	р@E
7while_simple_rnn_cell_6_biasadd_readvariableop_resource:@J
8while_simple_rnn_cell_6_matmul_1_readvariableop_resource:@@Ђ.while/simple_rnn_cell_6/BiasAdd/ReadVariableOpЂ-while/simple_rnn_cell_6/MatMul/ReadVariableOpЂ/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџр  Ї
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџр*
element_dtype0Ї
-while/simple_rnn_cell_6/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_6_matmul_readvariableop_resource_0*
_output_shapes
:	р@*
dtype0У
while/simple_rnn_cell_6/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@Є
.while/simple_rnn_cell_6/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_6_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0О
while/simple_rnn_cell_6/BiasAddBiasAdd(while/simple_rnn_cell_6/MatMul:product:06while/simple_rnn_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@Њ
/while/simple_rnn_cell_6/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0Њ
 while/simple_rnn_cell_6/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@Ќ
while/simple_rnn_cell_6/addAddV2(while/simple_rnn_cell_6/BiasAdd:output:0*while/simple_rnn_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@w
while/simple_rnn_cell_6/TanhTanhwhile/simple_rnn_cell_6/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Щ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_6/Tanh:y:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: }
while/Identity_4Identity while/simple_rnn_cell_6/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@п

while/NoOpNoOp/^while/simple_rnn_cell_6/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_6/MatMul/ReadVariableOp0^while/simple_rnn_cell_6/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_6_biasadd_readvariableop_resource9while_simple_rnn_cell_6_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_6_matmul_1_readvariableop_resource:while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_6_matmul_readvariableop_resource8while_simple_rnn_cell_6_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ@: : : : : 2`
.while/simple_rnn_cell_6/BiasAdd/ReadVariableOp.while/simple_rnn_cell_6/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_6/MatMul/ReadVariableOp-while/simple_rnn_cell_6/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ@:

_output_shapes
: :

_output_shapes
: 
	
ф
while_cond_1931098
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_1931098___redundant_placeholder05
1while_while_cond_1931098___redundant_placeholder15
1while_while_cond_1931098___redundant_placeholder25
1while_while_cond_1931098___redundant_placeholder35
1while_while_cond_1931098___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1: : : : :џџџџџџџџџ : :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
!
к
while_body_1928442
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
!while_simple_rnn_cell_6_1928464_0:	р@/
!while_simple_rnn_cell_6_1928466_0:@3
!while_simple_rnn_cell_6_1928468_0:@@
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
while_simple_rnn_cell_6_1928464:	р@-
while_simple_rnn_cell_6_1928466:@1
while_simple_rnn_cell_6_1928468:@@Ђ/while/simple_rnn_cell_6/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџр  Ї
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџр*
element_dtype0І
/while/simple_rnn_cell_6/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2!while_simple_rnn_cell_6_1928464_0!while_simple_rnn_cell_6_1928466_0!while_simple_rnn_cell_6_1928468_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ@:џџџџџџџџџ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_simple_rnn_cell_6_layer_call_and_return_conditional_losses_1928429с
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder8while/simple_rnn_cell_6/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity8while/simple_rnn_cell_6/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@~

while/NoOpNoOp0^while/simple_rnn_cell_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_simple_rnn_cell_6_1928464!while_simple_rnn_cell_6_1928464_0"D
while_simple_rnn_cell_6_1928466!while_simple_rnn_cell_6_1928466_0"D
while_simple_rnn_cell_6_1928468!while_simple_rnn_cell_6_1928468_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ@: : : : : 2b
/while/simple_rnn_cell_6/StatefulPartitionedCall/while/simple_rnn_cell_6/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ@:

_output_shapes
: :

_output_shapes
: 
н-
у
while_body_1931611
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ@*
element_dtype0
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ`{
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:џџџџџџџџџ`W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Е
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ`
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:џџџџџџџџџ`Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Л
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_splitr
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ Y
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ t
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ ]
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ o
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ k
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ U

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ l
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ P
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?k
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ c
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ h
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/add_3:z:0*
_output_shapes
: *
element_dtype0:щшвO
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ "4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?: : : : :џџџџџџџџџ : : :@`:`: `:`: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@`: 

_output_shapes
:`:$	 

_output_shapes

: `: 


_output_shapes
:`
п
Џ
while_cond_1929531
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1929531___redundant_placeholder05
1while_while_cond_1929531___redundant_placeholder15
1while_while_cond_1929531___redundant_placeholder25
1while_while_cond_1929531___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :џџџџџџџџџ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ@:

_output_shapes
: :

_output_shapes
:
п
Џ
while_cond_1928441
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1928441___redundant_placeholder05
1while_while_cond_1928441___redundant_placeholder15
1while_while_cond_1928441___redundant_placeholder25
1while_while_cond_1928441___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :џџџџџџџџџ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ@:

_output_shapes
: :

_output_shapes
:
Ћ=
Н
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_1929598

inputsC
0simple_rnn_cell_6_matmul_readvariableop_resource:	р@?
1simple_rnn_cell_6_biasadd_readvariableop_resource:@D
2simple_rnn_cell_6_matmul_1_readvariableop_resource:@@
identityЂ(simple_rnn_cell_6/BiasAdd/ReadVariableOpЂ'simple_rnn_cell_6/MatMul/ReadVariableOpЂ)simple_rnn_cell_6/MatMul_1/ReadVariableOpЂwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџрD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџр  р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџр*
shrink_axis_mask
'simple_rnn_cell_6/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_6_matmul_readvariableop_resource*
_output_shapes
:	р@*
dtype0
simple_rnn_cell_6/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
(simple_rnn_cell_6/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ќ
simple_rnn_cell_6/BiasAddBiasAdd"simple_rnn_cell_6/MatMul:product:00simple_rnn_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
)simple_rnn_cell_6/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_6_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0
simple_rnn_cell_6/MatMul_1MatMulzeros:output:01simple_rnn_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
simple_rnn_cell_6/addAddV2"simple_rnn_cell_6/BiasAdd:output:0$simple_rnn_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@k
simple_rnn_cell_6/TanhTanhsimple_rnn_cell_6/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : к
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_6_matmul_readvariableop_resource1simple_rnn_cell_6_biasadd_readvariableop_resource2simple_rnn_cell_6_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1929532*
condR
while_cond_1929531*8
output_shapes'
%: : : : :џџџџџџџџџ@: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   Т
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ@Я
NoOpNoOp)^simple_rnn_cell_6/BiasAdd/ReadVariableOp(^simple_rnn_cell_6/MatMul/ReadVariableOp*^simple_rnn_cell_6/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџр: : : 2T
(simple_rnn_cell_6/BiasAdd/ReadVariableOp(simple_rnn_cell_6/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_6/MatMul/ReadVariableOp'simple_rnn_cell_6/MatMul/ReadVariableOp2V
)simple_rnn_cell_6/MatMul_1/ReadVariableOp)simple_rnn_cell_6/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:џџџџџџџџџр
 
_user_specified_nameinputs
>
К
'__forward_gpu_gru_with_fallback_1931913

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dimc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџV
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:РS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Џ
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    Ю
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*H
_output_shapes6
4:џџџџџџџџџ :џџџџџџџџџ : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:О
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџ *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_1
ExpandDimsstrided_slice_0:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:џџџџџџџџџ c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџ I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"

cudnnrnn_0CudnnRNN:output:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:џџџџџџџџџ@:џџџџџџџџџ :@`: `:`*<
api_implements*(gru_f1b5063f-a134-4d67-872c-0c565cc5da6e*
api_preferred_deviceGPU*X
backward_function_name><__inference___backward_gpu_gru_with_fallback_1931778_1931914*
go_backwards( *

time_major( :S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias

ъ
N__inference_simple_rnn_cell_6_layer_call_and_return_conditional_losses_1928429

inputs

states1
matmul_readvariableop_resource:	р@-
biasadd_readvariableop_resource:@2
 matmul_1_readvariableop_resource:@@
identity

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	р@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@G
TanhTanhadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:џџџџџџџџџр:џџџџџџџџџ@: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџр
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_namestates

Й
.__inference_simple_rnn_6_layer_call_fn_1932021

inputs
unknown:	р@
	unknown_0:@
	unknown_1:@@
identityЂStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_1930592s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:џџџџџџџџџр: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџр
 
_user_specified_nameinputs
Ф8
б
simple_rnn_6_while_body_19309646
2simple_rnn_6_while_simple_rnn_6_while_loop_counter<
8simple_rnn_6_while_simple_rnn_6_while_maximum_iterations"
simple_rnn_6_while_placeholder$
 simple_rnn_6_while_placeholder_1$
 simple_rnn_6_while_placeholder_25
1simple_rnn_6_while_simple_rnn_6_strided_slice_1_0q
msimple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor_0X
Esimple_rnn_6_while_simple_rnn_cell_6_matmul_readvariableop_resource_0:	р@T
Fsimple_rnn_6_while_simple_rnn_cell_6_biasadd_readvariableop_resource_0:@Y
Gsimple_rnn_6_while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0:@@
simple_rnn_6_while_identity!
simple_rnn_6_while_identity_1!
simple_rnn_6_while_identity_2!
simple_rnn_6_while_identity_3!
simple_rnn_6_while_identity_43
/simple_rnn_6_while_simple_rnn_6_strided_slice_1o
ksimple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_6_tensorarrayunstack_tensorlistfromtensorV
Csimple_rnn_6_while_simple_rnn_cell_6_matmul_readvariableop_resource:	р@R
Dsimple_rnn_6_while_simple_rnn_cell_6_biasadd_readvariableop_resource:@W
Esimple_rnn_6_while_simple_rnn_cell_6_matmul_1_readvariableop_resource:@@Ђ;simple_rnn_6/while/simple_rnn_cell_6/BiasAdd/ReadVariableOpЂ:simple_rnn_6/while/simple_rnn_cell_6/MatMul/ReadVariableOpЂ<simple_rnn_6/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp
Dsimple_rnn_6/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџр  ш
6simple_rnn_6/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_6_while_placeholderMsimple_rnn_6/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџр*
element_dtype0С
:simple_rnn_6/while/simple_rnn_cell_6/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_6_while_simple_rnn_cell_6_matmul_readvariableop_resource_0*
_output_shapes
:	р@*
dtype0ъ
+simple_rnn_6/while/simple_rnn_cell_6/MatMulMatMul=simple_rnn_6/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_6/while/simple_rnn_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@О
;simple_rnn_6/while/simple_rnn_cell_6/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_6_while_simple_rnn_cell_6_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0х
,simple_rnn_6/while/simple_rnn_cell_6/BiasAddBiasAdd5simple_rnn_6/while/simple_rnn_cell_6/MatMul:product:0Csimple_rnn_6/while/simple_rnn_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@Ф
<simple_rnn_6/while/simple_rnn_cell_6/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_6_while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0б
-simple_rnn_6/while/simple_rnn_cell_6/MatMul_1MatMul simple_rnn_6_while_placeholder_2Dsimple_rnn_6/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@г
(simple_rnn_6/while/simple_rnn_cell_6/addAddV25simple_rnn_6/while/simple_rnn_cell_6/BiasAdd:output:07simple_rnn_6/while/simple_rnn_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@
)simple_rnn_6/while/simple_rnn_cell_6/TanhTanh,simple_rnn_6/while/simple_rnn_cell_6/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@§
7simple_rnn_6/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_6_while_placeholder_1simple_rnn_6_while_placeholder-simple_rnn_6/while/simple_rnn_cell_6/Tanh:y:0*
_output_shapes
: *
element_dtype0:щшвZ
simple_rnn_6/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
simple_rnn_6/while/addAddV2simple_rnn_6_while_placeholder!simple_rnn_6/while/add/y:output:0*
T0*
_output_shapes
: \
simple_rnn_6/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
simple_rnn_6/while/add_1AddV22simple_rnn_6_while_simple_rnn_6_while_loop_counter#simple_rnn_6/while/add_1/y:output:0*
T0*
_output_shapes
: 
simple_rnn_6/while/IdentityIdentitysimple_rnn_6/while/add_1:z:0^simple_rnn_6/while/NoOp*
T0*
_output_shapes
: 
simple_rnn_6/while/Identity_1Identity8simple_rnn_6_while_simple_rnn_6_while_maximum_iterations^simple_rnn_6/while/NoOp*
T0*
_output_shapes
: 
simple_rnn_6/while/Identity_2Identitysimple_rnn_6/while/add:z:0^simple_rnn_6/while/NoOp*
T0*
_output_shapes
: ­
simple_rnn_6/while/Identity_3IdentityGsimple_rnn_6/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_6/while/NoOp*
T0*
_output_shapes
: Є
simple_rnn_6/while/Identity_4Identity-simple_rnn_6/while/simple_rnn_cell_6/Tanh:y:0^simple_rnn_6/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@
simple_rnn_6/while/NoOpNoOp<^simple_rnn_6/while/simple_rnn_cell_6/BiasAdd/ReadVariableOp;^simple_rnn_6/while/simple_rnn_cell_6/MatMul/ReadVariableOp=^simple_rnn_6/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "C
simple_rnn_6_while_identity$simple_rnn_6/while/Identity:output:0"G
simple_rnn_6_while_identity_1&simple_rnn_6/while/Identity_1:output:0"G
simple_rnn_6_while_identity_2&simple_rnn_6/while/Identity_2:output:0"G
simple_rnn_6_while_identity_3&simple_rnn_6/while/Identity_3:output:0"G
simple_rnn_6_while_identity_4&simple_rnn_6/while/Identity_4:output:0"d
/simple_rnn_6_while_simple_rnn_6_strided_slice_11simple_rnn_6_while_simple_rnn_6_strided_slice_1_0"
Dsimple_rnn_6_while_simple_rnn_cell_6_biasadd_readvariableop_resourceFsimple_rnn_6_while_simple_rnn_cell_6_biasadd_readvariableop_resource_0"
Esimple_rnn_6_while_simple_rnn_cell_6_matmul_1_readvariableop_resourceGsimple_rnn_6_while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0"
Csimple_rnn_6_while_simple_rnn_cell_6_matmul_readvariableop_resourceEsimple_rnn_6_while_simple_rnn_cell_6_matmul_readvariableop_resource_0"м
ksimple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_6_tensorarrayunstack_tensorlistfromtensormsimple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ@: : : : : 2z
;simple_rnn_6/while/simple_rnn_cell_6/BiasAdd/ReadVariableOp;simple_rnn_6/while/simple_rnn_cell_6/BiasAdd/ReadVariableOp2x
:simple_rnn_6/while/simple_rnn_cell_6/MatMul/ReadVariableOp:simple_rnn_6/while/simple_rnn_cell_6/MatMul/ReadVariableOp2|
<simple_rnn_6/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp<simple_rnn_6/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ@:

_output_shapes
: :

_output_shapes
: 
	
ф
while_cond_1933703
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_1933703___redundant_placeholder05
1while_while_cond_1933703___redundant_placeholder15
1while_while_cond_1933703___redundant_placeholder25
1while_while_cond_1933703___redundant_placeholder35
1while_while_cond_1933703___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1: : : : :џџџџџџџџџ : :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
П

І
simple_rnn_6_while_cond_19309636
2simple_rnn_6_while_simple_rnn_6_while_loop_counter<
8simple_rnn_6_while_simple_rnn_6_while_maximum_iterations"
simple_rnn_6_while_placeholder$
 simple_rnn_6_while_placeholder_1$
 simple_rnn_6_while_placeholder_28
4simple_rnn_6_while_less_simple_rnn_6_strided_slice_1O
Ksimple_rnn_6_while_simple_rnn_6_while_cond_1930963___redundant_placeholder0O
Ksimple_rnn_6_while_simple_rnn_6_while_cond_1930963___redundant_placeholder1O
Ksimple_rnn_6_while_simple_rnn_6_while_cond_1930963___redundant_placeholder2O
Ksimple_rnn_6_while_simple_rnn_6_while_cond_1930963___redundant_placeholder3
simple_rnn_6_while_identity

simple_rnn_6/while/LessLesssimple_rnn_6_while_placeholder4simple_rnn_6_while_less_simple_rnn_6_strided_slice_1*
T0*
_output_shapes
: e
simple_rnn_6/while/IdentityIdentitysimple_rnn_6/while/Less:z:0*
T0
*
_output_shapes
: "C
simple_rnn_6_while_identity$simple_rnn_6/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :џџџџџџџџџ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ@:

_output_shapes
: :

_output_shapes
:
Є>
Є
 __inference_standard_gru_1929229

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3M
unstackUnpackbias*
T0* 
_output_shapes
:`:`*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*
shrink_axis_maskd
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:џџџџџџџџџ`h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ѓ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split^
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:џџџџџџџџџ`l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Љ
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ S
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:џџџџџџџџџ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ Q
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ў
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :џџџџџџџџџ : : :@`:`: `:`* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_1929139*
condR
while_cond_1929138*R
output_shapesA
?: : : : :џџџџџџџџџ : : :@`:`: `:`*
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ ]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:џџџџџџџџџ I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџ :@`: `:`*<
api_implements*(gru_7ffb1024-6f26-45b8-a31e-290afcdaacd2*
api_preferred_deviceCPU*
go_backwards( *

time_major( :\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
п
Џ
while_cond_1932386
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1932386___redundant_placeholder05
1while_while_cond_1932386___redundant_placeholder15
1while_while_cond_1932386___redundant_placeholder25
1while_while_cond_1932386___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :џџџџџџџџџ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ@:

_output_shapes
: :

_output_shapes
:
Щ

E__inference_conv1d_8_layer_call_and_return_conditional_losses_1931953

inputsA
+conv1d_expanddims_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂ"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: ­
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџ *
squeeze_dims

§џџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ 
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ф8
б
simple_rnn_6_while_body_19314766
2simple_rnn_6_while_simple_rnn_6_while_loop_counter<
8simple_rnn_6_while_simple_rnn_6_while_maximum_iterations"
simple_rnn_6_while_placeholder$
 simple_rnn_6_while_placeholder_1$
 simple_rnn_6_while_placeholder_25
1simple_rnn_6_while_simple_rnn_6_strided_slice_1_0q
msimple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor_0X
Esimple_rnn_6_while_simple_rnn_cell_6_matmul_readvariableop_resource_0:	р@T
Fsimple_rnn_6_while_simple_rnn_cell_6_biasadd_readvariableop_resource_0:@Y
Gsimple_rnn_6_while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0:@@
simple_rnn_6_while_identity!
simple_rnn_6_while_identity_1!
simple_rnn_6_while_identity_2!
simple_rnn_6_while_identity_3!
simple_rnn_6_while_identity_43
/simple_rnn_6_while_simple_rnn_6_strided_slice_1o
ksimple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_6_tensorarrayunstack_tensorlistfromtensorV
Csimple_rnn_6_while_simple_rnn_cell_6_matmul_readvariableop_resource:	р@R
Dsimple_rnn_6_while_simple_rnn_cell_6_biasadd_readvariableop_resource:@W
Esimple_rnn_6_while_simple_rnn_cell_6_matmul_1_readvariableop_resource:@@Ђ;simple_rnn_6/while/simple_rnn_cell_6/BiasAdd/ReadVariableOpЂ:simple_rnn_6/while/simple_rnn_cell_6/MatMul/ReadVariableOpЂ<simple_rnn_6/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp
Dsimple_rnn_6/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџр  ш
6simple_rnn_6/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_6_while_placeholderMsimple_rnn_6/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџр*
element_dtype0С
:simple_rnn_6/while/simple_rnn_cell_6/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_6_while_simple_rnn_cell_6_matmul_readvariableop_resource_0*
_output_shapes
:	р@*
dtype0ъ
+simple_rnn_6/while/simple_rnn_cell_6/MatMulMatMul=simple_rnn_6/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_6/while/simple_rnn_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@О
;simple_rnn_6/while/simple_rnn_cell_6/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_6_while_simple_rnn_cell_6_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype0х
,simple_rnn_6/while/simple_rnn_cell_6/BiasAddBiasAdd5simple_rnn_6/while/simple_rnn_cell_6/MatMul:product:0Csimple_rnn_6/while/simple_rnn_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@Ф
<simple_rnn_6/while/simple_rnn_cell_6/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_6_while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype0б
-simple_rnn_6/while/simple_rnn_cell_6/MatMul_1MatMul simple_rnn_6_while_placeholder_2Dsimple_rnn_6/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@г
(simple_rnn_6/while/simple_rnn_cell_6/addAddV25simple_rnn_6/while/simple_rnn_cell_6/BiasAdd:output:07simple_rnn_6/while/simple_rnn_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@
)simple_rnn_6/while/simple_rnn_cell_6/TanhTanh,simple_rnn_6/while/simple_rnn_cell_6/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@§
7simple_rnn_6/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_6_while_placeholder_1simple_rnn_6_while_placeholder-simple_rnn_6/while/simple_rnn_cell_6/Tanh:y:0*
_output_shapes
: *
element_dtype0:щшвZ
simple_rnn_6/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
simple_rnn_6/while/addAddV2simple_rnn_6_while_placeholder!simple_rnn_6/while/add/y:output:0*
T0*
_output_shapes
: \
simple_rnn_6/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
simple_rnn_6/while/add_1AddV22simple_rnn_6_while_simple_rnn_6_while_loop_counter#simple_rnn_6/while/add_1/y:output:0*
T0*
_output_shapes
: 
simple_rnn_6/while/IdentityIdentitysimple_rnn_6/while/add_1:z:0^simple_rnn_6/while/NoOp*
T0*
_output_shapes
: 
simple_rnn_6/while/Identity_1Identity8simple_rnn_6_while_simple_rnn_6_while_maximum_iterations^simple_rnn_6/while/NoOp*
T0*
_output_shapes
: 
simple_rnn_6/while/Identity_2Identitysimple_rnn_6/while/add:z:0^simple_rnn_6/while/NoOp*
T0*
_output_shapes
: ­
simple_rnn_6/while/Identity_3IdentityGsimple_rnn_6/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_6/while/NoOp*
T0*
_output_shapes
: Є
simple_rnn_6/while/Identity_4Identity-simple_rnn_6/while/simple_rnn_cell_6/Tanh:y:0^simple_rnn_6/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@
simple_rnn_6/while/NoOpNoOp<^simple_rnn_6/while/simple_rnn_cell_6/BiasAdd/ReadVariableOp;^simple_rnn_6/while/simple_rnn_cell_6/MatMul/ReadVariableOp=^simple_rnn_6/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "C
simple_rnn_6_while_identity$simple_rnn_6/while/Identity:output:0"G
simple_rnn_6_while_identity_1&simple_rnn_6/while/Identity_1:output:0"G
simple_rnn_6_while_identity_2&simple_rnn_6/while/Identity_2:output:0"G
simple_rnn_6_while_identity_3&simple_rnn_6/while/Identity_3:output:0"G
simple_rnn_6_while_identity_4&simple_rnn_6/while/Identity_4:output:0"d
/simple_rnn_6_while_simple_rnn_6_strided_slice_11simple_rnn_6_while_simple_rnn_6_strided_slice_1_0"
Dsimple_rnn_6_while_simple_rnn_cell_6_biasadd_readvariableop_resourceFsimple_rnn_6_while_simple_rnn_cell_6_biasadd_readvariableop_resource_0"
Esimple_rnn_6_while_simple_rnn_cell_6_matmul_1_readvariableop_resourceGsimple_rnn_6_while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0"
Csimple_rnn_6_while_simple_rnn_cell_6_matmul_readvariableop_resourceEsimple_rnn_6_while_simple_rnn_cell_6_matmul_readvariableop_resource_0"м
ksimple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_6_tensorarrayunstack_tensorlistfromtensormsimple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ@: : : : : 2z
;simple_rnn_6/while/simple_rnn_cell_6/BiasAdd/ReadVariableOp;simple_rnn_6/while/simple_rnn_cell_6/BiasAdd/ReadVariableOp2x
:simple_rnn_6/while/simple_rnn_cell_6/MatMul/ReadVariableOp:simple_rnn_6/while/simple_rnn_cell_6/MatMul/ReadVariableOp2|
<simple_rnn_6/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp<simple_rnn_6/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ@:

_output_shapes
: :

_output_shapes
: 
ў
к

<__inference___backward_gpu_gru_with_fallback_1931266_1931402
placeholder
placeholder_1
placeholder_2
placeholder_33
/gradients_expanddims_1_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:џџџџџџџџџ d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:џџџџџџџџџ `
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:џџџџџџџџџ O
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 
!gradients/ExpandDims_1_grad/ShapeShape/gradients_expanddims_1_grad_shape_strided_slice*
T0*
_output_shapes
:Њ
#gradients/ExpandDims_1_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:Є
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Х
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_1_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:џџџџџџџџџ }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџ *
shrink_axis_maska
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*K
_output_shapes9
7:џџџџџџџџџ@:џџџџџџџџџ : :РI*
rnn_modegru
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:а
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:Х
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ \
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:g
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: 
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::ц
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:щ
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
: щ
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
: щ
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: щ
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: ь
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: ь
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   Ѓ
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   Ѕ
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   Ѕ
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ѕ
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ѕ
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ѕ
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  h
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ё
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ё
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ё
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ѓ
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Є
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Є
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: 
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:З
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ 
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:З
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ 
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:З
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ 
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:З
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:З
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:З
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Р
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:@`
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

: `m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   `   Ё
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:`r
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ e

Identity_2Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes

:@`g

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes

: `h

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes

:`"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ : :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ : ::џџџџџџџџџ@:џџџџџџџџџ : :РI::џџџџџџџџџ : ::::::: : : *<
api_implements*(gru_7cdd8818-ca5e-4d7b-98a6-7015d31c0e59*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_gru_with_fallback_1931401*
go_backwards( *

time_major( :- )
'
_output_shapes
:џџџџџџџџџ :1-
+
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :1-
+
_output_shapes
:џџџџџџџџџ :1-
+
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
::1	-
+
_output_shapes
:џџџџџџџџџ@:1
-
+
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :!

_output_shapes	
:РI: 

_output_shapes
::-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ш	
і
E__inference_dense_29_layer_call_and_return_conditional_losses_1934047

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
л
Н
B__inference_gru_9_layer_call_and_return_conditional_losses_1929983

inputs.
read_readvariableop_resource:@`0
read_1_readvariableop_resource: `0
read_2_readvariableop_resource:`

identity_3ЂRead/ReadVariableOpЂRead_1/ReadVariableOpЂRead_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ p
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes

:@`*
dtype0Z
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes

:@`t
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes

: `*
dtype0^

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes

: `t
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes

:`*
dtype0^

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes

:`
PartitionedCallPartitionedCallinputszeros:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *S
_output_shapesA
?:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference_standard_gru_1929768i

Identity_3IdentityPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ 
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ@: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
>
К
'__forward_gpu_gru_with_fallback_1933628

inputs
init_h_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3
strided_slice
cudnnrnn

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
cudnnrnn_input_c

concat
transpose_perm

init_h
concat_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
split_2_split_dim
split_split_dim
split_1_split_dimc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџV
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:РS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Џ
concat_0ConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0U
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    Ю
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat_0:output:0*
T0*H
_output_shapes6
4:џџџџџџџџџ :џџџџџџџџџ : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:О
strided_slice_0StridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџ *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_1
ExpandDimsstrided_slice_0:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @`
IdentityIdentitystrided_slice_0:output:0*
T0*'
_output_shapes
:џџџџџџџџџ c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџ I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
concatconcat_0:output:0"#
concat_axisconcat/axis:output:0"
cudnnrnnCudnnRNN:output_h:0"

cudnnrnn_0CudnnRNN:output:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
cudnnrnn_input_cCudnnRNN/input_c:output:0"!

expanddimsExpandDims:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0")
strided_slicestrided_slice_0:output:0"
	transposetranspose_0:y:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0")
transpose_permtranspose/perm:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:џџџџџџџџџ@:џџџџџџџџџ :@`: `:`*<
api_implements*(gru_351bbcc5-99cd-49ef-abe4-0838c20dc796*
api_preferred_deviceGPU*X
backward_function_name><__inference___backward_gpu_gru_with_fallback_1933493_1933629*
go_backwards( *

time_major( :S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
	
ф
while_cond_1931610
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_1931610___redundant_placeholder05
1while_while_cond_1931610___redundant_placeholder15
1while_while_cond_1931610___redundant_placeholder25
1while_while_cond_1931610___redundant_placeholder35
1while_while_cond_1931610___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1: : : : :џџџџџџџџџ : :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
Ш	
і
E__inference_dense_28_layer_call_and_return_conditional_losses_1930001

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
ў
к

<__inference___backward_gpu_gru_with_fallback_1931778_1931914
placeholder
placeholder_1
placeholder_2
placeholder_33
/gradients_expanddims_1_grad_shape_strided_slice)
%gradients_squeeze_grad_shape_cudnnrnn/
+gradients_strided_slice_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn#
gradients_zeros_like_1_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims=
9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c3
/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h)
%gradients_concat_grad_mod_concat_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim
identity

identity_1

identity_2

identity_3

identity_4^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:џџџџџџџџџ d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:џџџџџџџџџ `
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:џџџџџџџџџ O
gradients/grad_ys_3Identityplaceholder_3*
T0*
_output_shapes
: 
!gradients/ExpandDims_1_grad/ShapeShape/gradients_expanddims_1_grad_shape_strided_slice*
T0*
_output_shapes
:Њ
#gradients/ExpandDims_1_grad/ReshapeReshapegradients/grad_ys_1:output:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ q
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
:Є
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Х
gradients/AddNAddNgradients/grad_ys_0:output:0,gradients/ExpandDims_1_grad/Reshape:output:0*
N*
T0*&
_class
loc:@gradients/grad_ys_0*'
_output_shapes
:џџџџџџџџџ }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/AddN:sum:0*
Index0*
T0*+
_output_shapes
:џџџџџџџџџ *
shrink_axis_maska
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
: g
gradients/zeros_like_1	ZerosLikegradients_zeros_like_1_cudnnrnn*
T0*
_output_shapes
:
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9gradients_cudnnrnn_grad_cudnnrnnbackprop_cudnnrnn_input_c/gradients_cudnnrnn_grad_cudnnrnnbackprop_concat+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnngradients_zeros_like_cudnnrnn6gradients/strided_slice_grad/StridedSliceGrad:output:0'gradients/Squeeze_grad/Reshape:output:0gradients/zeros_like:y:0gradients_zeros_like_1_cudnnrnn*
T0*K
_output_shapes9
7:џџџџџџџџџ@:џџџџџџџџџ : :РI*
rnn_modegru
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:а
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:Х
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ \
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:h
gradients/concat_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:g
gradients/concat_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: g
gradients/concat_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: h
gradients/concat_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: 
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0&gradients/concat_grad/Shape_2:output:0&gradients/concat_grad/Shape_3:output:0&gradients/concat_grad/Shape_4:output:0&gradients/concat_grad/Shape_5:output:0&gradients/concat_grad/Shape_6:output:0&gradients/concat_grad/Shape_7:output:0&gradients/concat_grad/Shape_8:output:0&gradients/concat_grad/Shape_9:output:0'gradients/concat_grad/Shape_10:output:0'gradients/concat_grad/Shape_11:output:0*
N*\
_output_shapesJ
H::::::::::::ц
gradients/concat_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:2&gradients/concat_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:3&gradients/concat_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:4&gradients/concat_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:ъ
gradients/concat_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:5&gradients/concat_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:щ
gradients/concat_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:6&gradients/concat_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes
: щ
gradients/concat_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:7&gradients/concat_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes
: щ
gradients/concat_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:8&gradients/concat_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: щ
gradients/concat_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0+gradients/concat_grad/ConcatOffset:offset:9&gradients/concat_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: ь
gradients/concat_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:10'gradients/concat_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: ь
gradients/concat_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0,gradients/concat_grad/ConcatOffset:offset:11'gradients/concat_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   Ѓ
 gradients/Reshape_1_grad/ReshapeReshape$gradients/concat_grad/Slice:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   Ѕ
 gradients/Reshape_2_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    @   Ѕ
 gradients/Reshape_3_grad/ReshapeReshape&gradients/concat_grad/Slice_2:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

: @o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ѕ
 gradients/Reshape_4_grad/ReshapeReshape&gradients/concat_grad/Slice_3:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ѕ
 gradients/Reshape_5_grad/ReshapeReshape&gradients/concat_grad/Slice_4:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        Ѕ
 gradients/Reshape_6_grad/ReshapeReshape&gradients/concat_grad/Slice_5:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  h
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ё
 gradients/Reshape_7_grad/ReshapeReshape&gradients/concat_grad/Slice_6:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ё
 gradients/Reshape_8_grad/ReshapeReshape&gradients/concat_grad/Slice_7:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ё
 gradients/Reshape_9_grad/ReshapeReshape&gradients/concat_grad/Slice_8:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Ѓ
!gradients/Reshape_10_grad/ReshapeReshape&gradients/concat_grad/Slice_9:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Є
!gradients/Reshape_11_grad/ReshapeReshape'gradients/concat_grad/Slice_10:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: Є
!gradients/Reshape_12_grad/ReshapeReshape'gradients/concat_grad/Slice_11:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: 
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:З
$gradients/transpose_1_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ 
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:З
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ 
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:З
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ 
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:З
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:З
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:З
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_7_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Р
gradients/split_grad/concatConcatV2(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes

:@`
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes

: `m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"   `   Ё
gradients/Reshape_grad/ReshapeReshape&gradients/split_2_grad/concat:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:`r
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ e

Identity_2Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes

:@`g

Identity_3Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes

: `h

Identity_4Identity'gradients/Reshape_grad/Reshape:output:0*
T0*
_output_shapes

:`"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ : :џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ : ::џџџџџџџџџ@:џџџџџџџџџ : :РI::џџџџџџџџџ : ::::::: : : *<
api_implements*(gru_f1b5063f-a134-4d67-872c-0c565cc5da6e*
api_preferred_deviceGPU*B
forward_function_name)'__forward_gpu_gru_with_fallback_1931913*
go_backwards( *

time_major( :- )
'
_output_shapes
:џџџџџџџџџ :1-
+
_output_shapes
:џџџџџџџџџ :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :1-
+
_output_shapes
:џџџџџџџџџ :1-
+
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
::1	-
+
_output_shapes
:џџџџџџџџџ@:1
-
+
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :!

_output_shapes	
:РI: 

_output_shapes
::-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
М
h
L__inference_repeat_vector_2_layer_call_and_return_conditional_losses_1931977

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :x

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџZ
stackConst*
_output_shapes
:*
dtype0*!
valueB"         p
TileTileExpandDims:output:0stack:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџb
IdentityIdentityTile:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџџџџџџџџџџ:X T
0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Е
Л
.__inference_simple_rnn_6_layer_call_fn_1931999
inputs_0
unknown:	р@
	unknown_0:@
	unknown_1:@@
identityЂStatefulPartitionedCallњ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_1928664|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџџџџџџџџџџр: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџр
"
_user_specified_name
inputs/0
Щ4
А
)__inference_gpu_gru_with_fallback_1931777

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :~
splitSplitsplit/split_dim:output:0kernel*
T0*2
_output_shapes 
:@ :@ :@ *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*2
_output_shapes 
:  :  :  *
	num_split`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџV
ReshapeReshapebiasReshape/shape:output:0*
T0*
_output_shapes	
:РS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
split_2Splitsplit_2/split_dim:output:0Reshape:output:0*
T0*8
_output_shapes&
$: : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:1transpose_1/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_1Reshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:0transpose_2/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_2Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: @[
	Reshape_3Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_4	Transposesplit_1:output:1transpose_4/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:a
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:2transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:[
	Reshape_7Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: [
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ъ
concatConcatV2Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0concat/axis:output:0*
N*
T0*
_output_shapes	
:РIU
CudnnRNN/input_cConst*
_output_shapes
: *
dtype0*
valueB
 *    Ъ
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0CudnnRNN/input_c:output:0concat:output:0*
T0*H
_output_shapes6
4:џџџџџџџџџ :џџџџџџџџџ : :*
rnn_modegruf
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_maskp
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:џџџџџџџџџ *
squeeze_dims
 R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :
ExpandDims_1
ExpandDimsstrided_slice:output:0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:џџџџџџџџџ c

Identity_1IdentityExpandDims_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:џџџџџџџџџ I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:џџџџџџџџџ@:џџџџџџџџџ :@`: `:`*<
api_implements*(gru_f1b5063f-a134-4d67-872c-0c565cc5da6e*
api_preferred_deviceGPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias
4
 
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_1928505

inputs,
simple_rnn_cell_6_1928430:	р@'
simple_rnn_cell_6_1928432:@+
simple_rnn_cell_6_1928434:@@
identityЂ)simple_rnn_cell_6/StatefulPartitionedCallЂwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџрD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџр  р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџр*
shrink_axis_maskы
)simple_rnn_cell_6/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_6_1928430simple_rnn_cell_6_1928432simple_rnn_cell_6_1928434*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ@:џџџџџџџџџ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_simple_rnn_cell_6_layer_call_and_return_conditional_losses_1928429n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_6_1928430simple_rnn_cell_6_1928432simple_rnn_cell_6_1928434*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_1928442*
condR
while_cond_1928441*8
output_shapes'
%: : : : :џџџџџџџџџ@: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   Ы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@z
NoOpNoOp*^simple_rnn_cell_6/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџџџџџџџџџџр: : : 2V
)simple_rnn_cell_6/StatefulPartitionedCall)simple_rnn_cell_6/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџр
 
_user_specified_nameinputs
н-
у
while_body_1928750
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ@*
element_dtype0
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ`{
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:џџџџџџџџџ`W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Е
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ`
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:џџџџџџџџџ`Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Л
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_splitr
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ Y
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ t
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ ]
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ o
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ k
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ U

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ l
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ P
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?k
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ c
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ h
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/add_3:z:0*
_output_shapes
: *
element_dtype0:щшвO
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ "4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?: : : : :џџџџџџџџџ : : :@`:`: `:`: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@`: 

_output_shapes
:`:$	 

_output_shapes

: `: 


_output_shapes
:`
П
з

J__inference_sequential_21_layer_call_and_return_conditional_losses_1931416

inputsJ
4conv1d_8_conv1d_expanddims_1_readvariableop_resource: 6
(conv1d_8_biasadd_readvariableop_resource: P
=simple_rnn_6_simple_rnn_cell_6_matmul_readvariableop_resource:	р@L
>simple_rnn_6_simple_rnn_cell_6_biasadd_readvariableop_resource:@Q
?simple_rnn_6_simple_rnn_cell_6_matmul_1_readvariableop_resource:@@4
"gru_9_read_readvariableop_resource:@`6
$gru_9_read_1_readvariableop_resource: `6
$gru_9_read_2_readvariableop_resource:`9
'dense_28_matmul_readvariableop_resource: 6
(dense_28_biasadd_readvariableop_resource:9
'dense_29_matmul_readvariableop_resource:6
(dense_29_biasadd_readvariableop_resource:
identityЂconv1d_8/BiasAdd/ReadVariableOpЂ+conv1d_8/Conv1D/ExpandDims_1/ReadVariableOpЂdense_28/BiasAdd/ReadVariableOpЂdense_28/MatMul/ReadVariableOpЂdense_29/BiasAdd/ReadVariableOpЂdense_29/MatMul/ReadVariableOpЂgru_9/Read/ReadVariableOpЂgru_9/Read_1/ReadVariableOpЂgru_9/Read_2/ReadVariableOpЂ5simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOpЂ4simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOpЂ6simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOpЂsimple_rnn_6/whilei
conv1d_8/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ
conv1d_8/Conv1D/ExpandDims
ExpandDimsinputs'conv1d_8/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџЄ
+conv1d_8/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_8_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype0b
 conv1d_8/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Л
conv1d_8/Conv1D/ExpandDims_1
ExpandDims3conv1d_8/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_8/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: Ш
conv1d_8/Conv1DConv2D#conv1d_8/Conv1D/ExpandDims:output:0%conv1d_8/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides

conv1d_8/Conv1D/SqueezeSqueezeconv1d_8/Conv1D:output:0*
T0*+
_output_shapes
:џџџџџџџџџ *
squeeze_dims

§џџџџџџџџ
conv1d_8/BiasAdd/ReadVariableOpReadVariableOp(conv1d_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv1d_8/BiasAddBiasAdd conv1d_8/Conv1D/Squeeze:output:0'conv1d_8/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ f
conv1d_8/ReluReluconv1d_8/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ `
flatten_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџр  
flatten_9/ReshapeReshapeconv1d_8/Relu:activations:0flatten_9/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџр`
repeat_vector_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Є
repeat_vector_2/ExpandDims
ExpandDimsflatten_9/Reshape:output:0'repeat_vector_2/ExpandDims/dim:output:0*
T0*,
_output_shapes
:џџџџџџџџџрj
repeat_vector_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"         
repeat_vector_2/TileTile#repeat_vector_2/ExpandDims:output:0repeat_vector_2/stack:output:0*
T0*,
_output_shapes
:џџџџџџџџџр_
simple_rnn_6/ShapeShaperepeat_vector_2/Tile:output:0*
T0*
_output_shapes
:j
 simple_rnn_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"simple_rnn_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"simple_rnn_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
simple_rnn_6/strided_sliceStridedSlicesimple_rnn_6/Shape:output:0)simple_rnn_6/strided_slice/stack:output:0+simple_rnn_6/strided_slice/stack_1:output:0+simple_rnn_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
simple_rnn_6/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@
simple_rnn_6/zeros/packedPack#simple_rnn_6/strided_slice:output:0$simple_rnn_6/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
simple_rnn_6/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
simple_rnn_6/zerosFill"simple_rnn_6/zeros/packed:output:0!simple_rnn_6/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@p
simple_rnn_6/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
simple_rnn_6/transpose	Transposerepeat_vector_2/Tile:output:0$simple_rnn_6/transpose/perm:output:0*
T0*,
_output_shapes
:џџџџџџџџџр^
simple_rnn_6/Shape_1Shapesimple_rnn_6/transpose:y:0*
T0*
_output_shapes
:l
"simple_rnn_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
simple_rnn_6/strided_slice_1StridedSlicesimple_rnn_6/Shape_1:output:0+simple_rnn_6/strided_slice_1/stack:output:0-simple_rnn_6/strided_slice_1/stack_1:output:0-simple_rnn_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(simple_rnn_6/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџл
simple_rnn_6/TensorArrayV2TensorListReserve1simple_rnn_6/TensorArrayV2/element_shape:output:0%simple_rnn_6/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
Bsimple_rnn_6/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџр  
4simple_rnn_6/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_6/transpose:y:0Ksimple_rnn_6/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвl
"simple_rnn_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ћ
simple_rnn_6/strided_slice_2StridedSlicesimple_rnn_6/transpose:y:0+simple_rnn_6/strided_slice_2/stack:output:0-simple_rnn_6/strided_slice_2/stack_1:output:0-simple_rnn_6/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџр*
shrink_axis_maskГ
4simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOpReadVariableOp=simple_rnn_6_simple_rnn_cell_6_matmul_readvariableop_resource*
_output_shapes
:	р@*
dtype0Ц
%simple_rnn_6/simple_rnn_cell_6/MatMulMatMul%simple_rnn_6/strided_slice_2:output:0<simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@А
5simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_6_simple_rnn_cell_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0г
&simple_rnn_6/simple_rnn_cell_6/BiasAddBiasAdd/simple_rnn_6/simple_rnn_cell_6/MatMul:product:0=simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@Ж
6simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_6_simple_rnn_cell_6_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype0Р
'simple_rnn_6/simple_rnn_cell_6/MatMul_1MatMulsimple_rnn_6/zeros:output:0>simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@С
"simple_rnn_6/simple_rnn_cell_6/addAddV2/simple_rnn_6/simple_rnn_cell_6/BiasAdd:output:01simple_rnn_6/simple_rnn_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ@
#simple_rnn_6/simple_rnn_cell_6/TanhTanh&simple_rnn_6/simple_rnn_cell_6/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@{
*simple_rnn_6/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   п
simple_rnn_6/TensorArrayV2_1TensorListReserve3simple_rnn_6/TensorArrayV2_1/element_shape:output:0%simple_rnn_6/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвS
simple_rnn_6/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%simple_rnn_6/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџa
simple_rnn_6/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
simple_rnn_6/whileWhile(simple_rnn_6/while/loop_counter:output:0.simple_rnn_6/while/maximum_iterations:output:0simple_rnn_6/time:output:0%simple_rnn_6/TensorArrayV2_1:handle:0simple_rnn_6/zeros:output:0%simple_rnn_6/strided_slice_1:output:0Dsimple_rnn_6/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_6_simple_rnn_cell_6_matmul_readvariableop_resource>simple_rnn_6_simple_rnn_cell_6_biasadd_readvariableop_resource?simple_rnn_6_simple_rnn_cell_6_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ@: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *+
body#R!
simple_rnn_6_while_body_1930964*+
cond#R!
simple_rnn_6_while_cond_1930963*8
output_shapes'
%: : : : :џџџџџџџџџ@: : : : : *
parallel_iterations 
=simple_rnn_6/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   щ
/simple_rnn_6/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_6/while:output:3Fsimple_rnn_6/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ@*
element_dtype0u
"simple_rnn_6/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџn
$simple_rnn_6/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_6/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ш
simple_rnn_6/strided_slice_3StridedSlice8simple_rnn_6/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_6/strided_slice_3/stack:output:0-simple_rnn_6/strided_slice_3/stack_1:output:0-simple_rnn_6/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*
shrink_axis_maskr
simple_rnn_6/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Н
simple_rnn_6/transpose_1	Transpose8simple_rnn_6/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_6/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@W
gru_9/ShapeShapesimple_rnn_6/transpose_1:y:0*
T0*
_output_shapes
:c
gru_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:я
gru_9/strided_sliceStridedSlicegru_9/Shape:output:0"gru_9/strided_slice/stack:output:0$gru_9/strided_slice/stack_1:output:0$gru_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
gru_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 
gru_9/zeros/packedPackgru_9/strided_slice:output:0gru_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:V
gru_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
gru_9/zerosFillgru_9/zeros/packed:output:0gru_9/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ |
gru_9/Read/ReadVariableOpReadVariableOp"gru_9_read_readvariableop_resource*
_output_shapes

:@`*
dtype0f
gru_9/IdentityIdentity!gru_9/Read/ReadVariableOp:value:0*
T0*
_output_shapes

:@`
gru_9/Read_1/ReadVariableOpReadVariableOp$gru_9_read_1_readvariableop_resource*
_output_shapes

: `*
dtype0j
gru_9/Identity_1Identity#gru_9/Read_1/ReadVariableOp:value:0*
T0*
_output_shapes

: `
gru_9/Read_2/ReadVariableOpReadVariableOp$gru_9_read_2_readvariableop_resource*
_output_shapes

:`*
dtype0j
gru_9/Identity_2Identity#gru_9/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes

:`П
gru_9/PartitionedCallPartitionedCallsimple_rnn_6/transpose_1:y:0gru_9/zeros:output:0gru_9/Identity:output:0gru_9/Identity_1:output:0gru_9/Identity_2:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *S
_output_shapesA
?:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference_standard_gru_1931189
dense_28/MatMul/ReadVariableOpReadVariableOp'dense_28_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
dense_28/MatMulMatMulgru_9/PartitionedCall:output:0&dense_28/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_28/BiasAddBiasAdddense_28/MatMul:product:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_29/MatMul/ReadVariableOpReadVariableOp'dense_29_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_29/MatMulMatMuldense_28/BiasAdd:output:0&dense_29/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_29/BiasAddBiasAdddense_29/MatMul:product:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџh
IdentityIdentitydense_29/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџБ
NoOpNoOp ^conv1d_8/BiasAdd/ReadVariableOp,^conv1d_8/Conv1D/ExpandDims_1/ReadVariableOp ^dense_28/BiasAdd/ReadVariableOp^dense_28/MatMul/ReadVariableOp ^dense_29/BiasAdd/ReadVariableOp^dense_29/MatMul/ReadVariableOp^gru_9/Read/ReadVariableOp^gru_9/Read_1/ReadVariableOp^gru_9/Read_2/ReadVariableOp6^simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOp5^simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOp7^simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOp^simple_rnn_6/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ: : : : : : : : : : : : 2B
conv1d_8/BiasAdd/ReadVariableOpconv1d_8/BiasAdd/ReadVariableOp2Z
+conv1d_8/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_8/Conv1D/ExpandDims_1/ReadVariableOp2B
dense_28/BiasAdd/ReadVariableOpdense_28/BiasAdd/ReadVariableOp2@
dense_28/MatMul/ReadVariableOpdense_28/MatMul/ReadVariableOp2B
dense_29/BiasAdd/ReadVariableOpdense_29/BiasAdd/ReadVariableOp2@
dense_29/MatMul/ReadVariableOpdense_29/MatMul/ReadVariableOp26
gru_9/Read/ReadVariableOpgru_9/Read/ReadVariableOp2:
gru_9/Read_1/ReadVariableOpgru_9/Read_1/ReadVariableOp2:
gru_9/Read_2/ReadVariableOpgru_9/Read_2/ReadVariableOp2n
5simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOp5simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOp2l
4simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOp4simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOp2p
6simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOp6simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOp2(
simple_rnn_6/whilesimple_rnn_6/while:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
е"
С
J__inference_sequential_21_layer_call_and_return_conditional_losses_1930683

inputs&
conv1d_8_1930651: 
conv1d_8_1930653: '
simple_rnn_6_1930658:	р@"
simple_rnn_6_1930660:@&
simple_rnn_6_1930662:@@
gru_9_1930665:@`
gru_9_1930667: `
gru_9_1930669:`"
dense_28_1930672: 
dense_28_1930674:"
dense_29_1930677:
dense_29_1930679:
identityЂ conv1d_8/StatefulPartitionedCallЂ dense_28/StatefulPartitionedCallЂ dense_29/StatefulPartitionedCallЂgru_9/StatefulPartitionedCallЂ$simple_rnn_6/StatefulPartitionedCallї
 conv1d_8/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_8_1930651conv1d_8_1930653*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv1d_8_layer_call_and_return_conditional_losses_1929476п
flatten_9/PartitionedCallPartitionedCall)conv1d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџр* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_9_layer_call_and_return_conditional_losses_1929488ш
repeat_vector_2/PartitionedCallPartitionedCall"flatten_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџр* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_repeat_vector_2_layer_call_and_return_conditional_losses_1928378С
$simple_rnn_6/StatefulPartitionedCallStatefulPartitionedCall(repeat_vector_2/PartitionedCall:output:0simple_rnn_6_1930658simple_rnn_6_1930660simple_rnn_6_1930662*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_1930592
gru_9/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_6/StatefulPartitionedCall:output:0gru_9_1930665gru_9_1930667gru_9_1930669*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_gru_9_layer_call_and_return_conditional_losses_1930462
 dense_28/StatefulPartitionedCallStatefulPartitionedCall&gru_9/StatefulPartitionedCall:output:0dense_28_1930672dense_28_1930674*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_28_layer_call_and_return_conditional_losses_1930001
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0dense_29_1930677dense_29_1930679*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_29_layer_call_and_return_conditional_losses_1930017x
IdentityIdentity)dense_29/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџі
NoOpNoOp!^conv1d_8/StatefulPartitionedCall!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall^gru_9/StatefulPartitionedCall%^simple_rnn_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:џџџџџџџџџ: : : : : : : : : : : : 2D
 conv1d_8/StatefulPartitionedCall conv1d_8/StatefulPartitionedCall2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall2>
gru_9/StatefulPartitionedCallgru_9/StatefulPartitionedCall2L
$simple_rnn_6/StatefulPartitionedCall$simple_rnn_6/StatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ф

*__inference_dense_29_layer_call_fn_1934037

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_29_layer_call_and_return_conditional_losses_1930017o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
і
Е
'__inference_gru_9_layer_call_fn_1932497

inputs
unknown:@`
	unknown_0: `
	unknown_1:`
identityЂStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_gru_9_layer_call_and_return_conditional_losses_1930462o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
	
ф
while_cond_1929138
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_1929138___redundant_placeholder05
1while_while_cond_1929138___redundant_placeholder15
1while_while_cond_1929138___redundant_placeholder25
1while_while_cond_1929138___redundant_placeholder35
1while_while_cond_1929138___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1: : : : :џџџџџџџџџ : :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
Г^
§
 __inference__traced_save_1934267
file_prefix.
*savev2_conv1d_8_kernel_read_readvariableop,
(savev2_conv1d_8_bias_read_readvariableop.
*savev2_dense_28_kernel_read_readvariableop,
(savev2_dense_28_bias_read_readvariableop.
*savev2_dense_29_kernel_read_readvariableop,
(savev2_dense_29_bias_read_readvariableopD
@savev2_simple_rnn_6_simple_rnn_cell_6_kernel_read_readvariableopN
Jsavev2_simple_rnn_6_simple_rnn_cell_6_recurrent_kernel_read_readvariableopB
>savev2_simple_rnn_6_simple_rnn_cell_6_bias_read_readvariableop6
2savev2_gru_9_gru_cell_9_kernel_read_readvariableop@
<savev2_gru_9_gru_cell_9_recurrent_kernel_read_readvariableop4
0savev2_gru_9_gru_cell_9_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_conv1d_8_kernel_m_read_readvariableop3
/savev2_adam_conv1d_8_bias_m_read_readvariableop5
1savev2_adam_dense_28_kernel_m_read_readvariableop3
/savev2_adam_dense_28_bias_m_read_readvariableop5
1savev2_adam_dense_29_kernel_m_read_readvariableop3
/savev2_adam_dense_29_bias_m_read_readvariableopK
Gsavev2_adam_simple_rnn_6_simple_rnn_cell_6_kernel_m_read_readvariableopU
Qsavev2_adam_simple_rnn_6_simple_rnn_cell_6_recurrent_kernel_m_read_readvariableopI
Esavev2_adam_simple_rnn_6_simple_rnn_cell_6_bias_m_read_readvariableop=
9savev2_adam_gru_9_gru_cell_9_kernel_m_read_readvariableopG
Csavev2_adam_gru_9_gru_cell_9_recurrent_kernel_m_read_readvariableop;
7savev2_adam_gru_9_gru_cell_9_bias_m_read_readvariableop5
1savev2_adam_conv1d_8_kernel_v_read_readvariableop3
/savev2_adam_conv1d_8_bias_v_read_readvariableop5
1savev2_adam_dense_28_kernel_v_read_readvariableop3
/savev2_adam_dense_28_bias_v_read_readvariableop5
1savev2_adam_dense_29_kernel_v_read_readvariableop3
/savev2_adam_dense_29_bias_v_read_readvariableopK
Gsavev2_adam_simple_rnn_6_simple_rnn_cell_6_kernel_v_read_readvariableopU
Qsavev2_adam_simple_rnn_6_simple_rnn_cell_6_recurrent_kernel_v_read_readvariableopI
Esavev2_adam_simple_rnn_6_simple_rnn_cell_6_bias_v_read_readvariableop=
9savev2_adam_gru_9_gru_cell_9_kernel_v_read_readvariableopG
Csavev2_adam_gru_9_gru_cell_9_recurrent_kernel_v_read_readvariableop;
7savev2_adam_gru_9_gru_cell_9_bias_v_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*О
valueДBБ.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЩ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Й
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv1d_8_kernel_read_readvariableop(savev2_conv1d_8_bias_read_readvariableop*savev2_dense_28_kernel_read_readvariableop(savev2_dense_28_bias_read_readvariableop*savev2_dense_29_kernel_read_readvariableop(savev2_dense_29_bias_read_readvariableop@savev2_simple_rnn_6_simple_rnn_cell_6_kernel_read_readvariableopJsavev2_simple_rnn_6_simple_rnn_cell_6_recurrent_kernel_read_readvariableop>savev2_simple_rnn_6_simple_rnn_cell_6_bias_read_readvariableop2savev2_gru_9_gru_cell_9_kernel_read_readvariableop<savev2_gru_9_gru_cell_9_recurrent_kernel_read_readvariableop0savev2_gru_9_gru_cell_9_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_conv1d_8_kernel_m_read_readvariableop/savev2_adam_conv1d_8_bias_m_read_readvariableop1savev2_adam_dense_28_kernel_m_read_readvariableop/savev2_adam_dense_28_bias_m_read_readvariableop1savev2_adam_dense_29_kernel_m_read_readvariableop/savev2_adam_dense_29_bias_m_read_readvariableopGsavev2_adam_simple_rnn_6_simple_rnn_cell_6_kernel_m_read_readvariableopQsavev2_adam_simple_rnn_6_simple_rnn_cell_6_recurrent_kernel_m_read_readvariableopEsavev2_adam_simple_rnn_6_simple_rnn_cell_6_bias_m_read_readvariableop9savev2_adam_gru_9_gru_cell_9_kernel_m_read_readvariableopCsavev2_adam_gru_9_gru_cell_9_recurrent_kernel_m_read_readvariableop7savev2_adam_gru_9_gru_cell_9_bias_m_read_readvariableop1savev2_adam_conv1d_8_kernel_v_read_readvariableop/savev2_adam_conv1d_8_bias_v_read_readvariableop1savev2_adam_dense_28_kernel_v_read_readvariableop/savev2_adam_dense_28_bias_v_read_readvariableop1savev2_adam_dense_29_kernel_v_read_readvariableop/savev2_adam_dense_29_bias_v_read_readvariableopGsavev2_adam_simple_rnn_6_simple_rnn_cell_6_kernel_v_read_readvariableopQsavev2_adam_simple_rnn_6_simple_rnn_cell_6_recurrent_kernel_v_read_readvariableopEsavev2_adam_simple_rnn_6_simple_rnn_cell_6_bias_v_read_readvariableop9savev2_adam_gru_9_gru_cell_9_kernel_v_read_readvariableopCsavev2_adam_gru_9_gru_cell_9_recurrent_kernel_v_read_readvariableop7savev2_adam_gru_9_gru_cell_9_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *<
dtypes2
02.	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*ђ
_input_shapesр
н: : : : ::::	р@:@@:@:@`: `:`: : : : : : : : : : : : ::::	р@:@@:@:@`: `:`: : : ::::	р@:@@:@:@`: `:`: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
: : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::%!

_output_shapes
:	р@:$ 

_output_shapes

:@@: 	

_output_shapes
:@:$
 

_output_shapes

:@`:$ 

_output_shapes

: `:$ 

_output_shapes

:`:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :($
"
_output_shapes
: : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::%!

_output_shapes
:	р@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@`:$  

_output_shapes

: `:$! 

_output_shapes

:`:("$
"
_output_shapes
: : #

_output_shapes
: :$$ 

_output_shapes

: : %

_output_shapes
::$& 

_output_shapes

:: '

_output_shapes
::%(!

_output_shapes
:	р@:$) 

_output_shapes

:@@: *

_output_shapes
:@:$+ 

_output_shapes

:@`:$, 

_output_shapes

: `:$- 

_output_shapes

:`:.

_output_shapes
: 
	
ф
while_cond_1928749
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_1928749___redundant_placeholder05
1while_while_cond_1928749___redundant_placeholder15
1while_while_cond_1928749___redundant_placeholder25
1while_while_cond_1928749___redundant_placeholder35
1while_while_cond_1928749___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1: : : : :џџџџџџџџџ : :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
н-
у
while_body_1931099
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0
while_biasadd_unstack_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_1_unstack_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel
while_biasadd_unstack#
while_matmul_1_recurrent_kernel
while_biasadd_1_unstack
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ@*
element_dtype0
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ`{
while/BiasAddBiasAddwhile/MatMul:product:0while_biasadd_unstack_0*
T0*'
_output_shapes
:џџџџџџџџџ`W
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Е
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*'
_output_shapes
:џџџџџџџџџ`
while/BiasAdd_1BiasAddwhile/MatMul_1:product:0while_biasadd_1_unstack_0*
T0*'
_output_shapes
:џџџџџџџџџ`Y
while/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Л
while/split_1Split while/split_1/split_dim:output:0while/BiasAdd_1:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_splitr
	while/addAddV2while/split:output:0while/split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ Y
while/SigmoidSigmoidwhile/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ t
while/add_1AddV2while/split:output:1while/split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ ]
while/Sigmoid_1Sigmoidwhile/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ o
	while/mulMulwhile/Sigmoid_1:y:0while/split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ k
while/add_2AddV2while/split:output:2while/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ U

while/TanhTanhwhile/add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ l
while/mul_1Mulwhile/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:џџџџџџџџџ P
while/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?k
	while/subSubwhile/sub/x:output:0while/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ c
while/mul_2Mulwhile/sub:z:0while/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ h
while/add_3AddV2while/mul_1:z:0while/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/add_3:z:0*
_output_shapes
: *
element_dtype0:щшвO
while/add_4/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_4AddV2while_placeholderwhile/add_4/y:output:0*
T0*
_output_shapes
: O
while/add_5/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_5AddV2while_while_loop_counterwhile/add_5/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_5:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_4:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/add_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ "4
while_biasadd_1_unstackwhile_biasadd_1_unstack_0"0
while_biasadd_unstackwhile_biasadd_unstack_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?: : : : :џџџџџџџџџ : : :@`:`: `:`: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@`: 

_output_shapes
:`:$	 

_output_shapes

: `: 


_output_shapes
:`
	
ф
while_cond_1930156
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice5
1while_while_cond_1930156___redundant_placeholder05
1while_while_cond_1930156___redundant_placeholder15
1while_while_cond_1930156___redundant_placeholder25
1while_while_cond_1930156___redundant_placeholder35
1while_while_cond_1930156___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1: : : : :џџџџџџџџџ : :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::	

_output_shapes
::


_output_shapes
:
п
Џ
while_cond_1932278
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_15
1while_while_cond_1932278___redundant_placeholder05
1while_while_cond_1932278___redundant_placeholder15
1while_while_cond_1932278___redundant_placeholder25
1while_while_cond_1932278___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :џџџџџџџџџ@: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:џџџџџџџџџ@:

_output_shapes
: :

_output_shapes
:
Ф

*__inference_dense_28_layer_call_fn_1934018

inputs
unknown: 
	unknown_0:
identityЂStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_28_layer_call_and_return_conditional_losses_1930001o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
>
Є
 __inference_standard_gru_1931189

inputs

init_h

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3M
unstackUnpackbias*
T0* 
_output_shapes
:`:`*	
numc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџВ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ@*
shrink_axis_maskd
MatMulMatMulstrided_slice_1:output:0kernel*
T0*'
_output_shapes
:џџџџџџџџџ`h
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ѓ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split^
MatMul_1MatMulinit_hrecurrent_kernel*
T0*'
_output_shapes
:џџџџџџџџџ`l
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:џџџџџџџџџ`S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Љ
split_1Splitsplit_1/split_dim:output:0BiasAdd_1:output:0*
T0*M
_output_shapes;
9:џџџџџџџџџ :џџџџџџџџџ :џџџџџџџџџ *
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:џџџџџџџџџ Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:џџџџџџџџџ Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ I
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ S
mul_1MulSigmoid:y:0init_h*
T0*'
_output_shapes
:џџџџџџџџџ J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ Q
mul_2Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Х
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ў
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelunstack:output:0recurrent_kernelunstack:output:1*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :џџџџџџџџџ : : :@`:`: `:`* 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_1931099*
condR
while_cond_1931098*R
output_shapesA
?: : : : :џџџџџџџџџ : : :@`:`: `:`*
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ    ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ *
element_dtype0*
num_elementsh
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  ?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ ]

Identity_1Identitytranspose_1:y:0*
T0*+
_output_shapes
:џџџџџџџџџ X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:џџџџџџџџџ I

Identity_3Identityruntime:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:џџџџџџџџџ@:џџџџџџџџџ :@`: `:`*<
api_implements*(gru_7cdd8818-ca5e-4d7b-98a6-7015d31c0e59*
api_preferred_deviceCPU*
go_backwards( *

time_major( :S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinit_h:FB

_output_shapes

:@`
 
_user_specified_namekernel:PL

_output_shapes

: `
*
_user_specified_namerecurrent_kernel:D@

_output_shapes

:`

_user_specified_namebias"Е	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Н
serving_defaultЉ
M
conv1d_8_input;
 serving_default_conv1d_8_input:0џџџџџџџџџ<
dense_290
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:
У
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
	variables
	trainable_variables

regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
н
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op"
_tf_keras_layer
Ѕ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses"
_tf_keras_layer
У
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses
,cell
-
state_spec"
_tf_keras_rnn_layer
к
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses
4_random_generator
5cell
6
state_spec"
_tf_keras_rnn_layer
Л
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses

=kernel
>bias"
_tf_keras_layer
Л
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses

Ekernel
Fbias"
_tf_keras_layer
v
0
1
G2
H3
I4
J5
K6
L7
=8
>9
E10
F11"
trackable_list_wrapper
v
0
1
G2
H3
I4
J5
K6
L7
=8
>9
E10
F11"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
	trainable_variables

regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ё
Rtrace_0
Strace_1
Ttrace_2
Utrace_32
/__inference_sequential_21_layer_call_fn_1930051
/__inference_sequential_21_layer_call_fn_1930875
/__inference_sequential_21_layer_call_fn_1930904
/__inference_sequential_21_layer_call_fn_1930739П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zRtrace_0zStrace_1zTtrace_2zUtrace_3
н
Vtrace_0
Wtrace_1
Xtrace_2
Ytrace_32ђ
J__inference_sequential_21_layer_call_and_return_conditional_losses_1931416
J__inference_sequential_21_layer_call_and_return_conditional_losses_1931928
J__inference_sequential_21_layer_call_and_return_conditional_losses_1930774
J__inference_sequential_21_layer_call_and_return_conditional_losses_1930809П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zVtrace_0zWtrace_1zXtrace_2zYtrace_3
дBб
"__inference__wrapped_model_1928366conv1d_8_input"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
У
Ziter

[beta_1

\beta_2
	]decay
^learning_ratemХmЦ=mЧ>mШEmЩFmЪGmЫHmЬImЭJmЮKmЯLmаvбvв=vг>vдEvеFvжGvзHvиIvйJvкKvлLvм"
	optimizer
,
_serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ю
etrace_02б
*__inference_conv1d_8_layer_call_fn_1931937Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zetrace_0

ftrace_02ь
E__inference_conv1d_8_layer_call_and_return_conditional_losses_1931953Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zftrace_0
%:# 2conv1d_8/kernel
: 2conv1d_8/bias
Д2БЎ
ЃВ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
я
ltrace_02в
+__inference_flatten_9_layer_call_fn_1931958Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zltrace_0

mtrace_02э
F__inference_flatten_9_layer_call_and_return_conditional_losses_1931964Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zmtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
ѕ
strace_02и
1__inference_repeat_vector_2_layer_call_fn_1931969Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zstrace_0

ttrace_02ѓ
L__inference_repeat_vector_2_layer_call_and_return_conditional_losses_1931977Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zttrace_0
5
G0
H1
I2"
trackable_list_wrapper
5
G0
H1
I2"
trackable_list_wrapper
 "
trackable_list_wrapper
Й

ustates
vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object

{trace_0
|trace_1
}trace_2
~trace_32
.__inference_simple_rnn_6_layer_call_fn_1931988
.__inference_simple_rnn_6_layer_call_fn_1931999
.__inference_simple_rnn_6_layer_call_fn_1932010
.__inference_simple_rnn_6_layer_call_fn_1932021д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z{trace_0z|trace_1z}trace_2z~trace_3
є
trace_0
trace_1
trace_2
trace_32
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_1932129
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_1932237
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_1932345
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_1932453д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1ztrace_2ztrace_3
я
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator

Gkernel
Hrecurrent_kernel
Ibias"
_tf_keras_layer
 "
trackable_list_wrapper
5
J0
K1
L2"
trackable_list_wrapper
5
J0
K1
L2"
trackable_list_wrapper
 "
trackable_list_wrapper
П
states
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
ю
trace_0
trace_1
trace_2
trace_32ћ
'__inference_gru_9_layer_call_fn_1932464
'__inference_gru_9_layer_call_fn_1932475
'__inference_gru_9_layer_call_fn_1932486
'__inference_gru_9_layer_call_fn_1932497д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1ztrace_2ztrace_3
к
trace_0
trace_1
trace_2
trace_32ч
B__inference_gru_9_layer_call_and_return_conditional_losses_1932875
B__inference_gru_9_layer_call_and_return_conditional_losses_1933253
B__inference_gru_9_layer_call_and_return_conditional_losses_1933631
B__inference_gru_9_layer_call_and_return_conditional_losses_1934009д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1ztrace_2ztrace_3
"
_generic_user_object
я
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator

Jkernel
Krecurrent_kernel
Lbias"
_tf_keras_layer
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
 layers
Ёmetrics
 Ђlayer_regularization_losses
Ѓlayer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
№
Єtrace_02б
*__inference_dense_28_layer_call_fn_1934018Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЄtrace_0

Ѕtrace_02ь
E__inference_dense_28_layer_call_and_return_conditional_losses_1934028Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЅtrace_0
!: 2dense_28/kernel
:2dense_28/bias
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Іnon_trainable_variables
Їlayers
Јmetrics
 Љlayer_regularization_losses
Њlayer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
№
Ћtrace_02б
*__inference_dense_29_layer_call_fn_1934037Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЋtrace_0

Ќtrace_02ь
E__inference_dense_29_layer_call_and_return_conditional_losses_1934047Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЌtrace_0
!:2dense_29/kernel
:2dense_29/bias
8:6	р@2%simple_rnn_6/simple_rnn_cell_6/kernel
A:?@@2/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel
1:/@2#simple_rnn_6/simple_rnn_cell_6/bias
):'@`2gru_9/gru_cell_9/kernel
3:1 `2!gru_9/gru_cell_9/recurrent_kernel
':%`2gru_9/gru_cell_9/bias
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
0
­0
Ў1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
/__inference_sequential_21_layer_call_fn_1930051conv1d_8_input"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B§
/__inference_sequential_21_layer_call_fn_1930875inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B§
/__inference_sequential_21_layer_call_fn_1930904inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
/__inference_sequential_21_layer_call_fn_1930739conv1d_8_input"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
J__inference_sequential_21_layer_call_and_return_conditional_losses_1931416inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
J__inference_sequential_21_layer_call_and_return_conditional_losses_1931928inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЃB 
J__inference_sequential_21_layer_call_and_return_conditional_losses_1930774conv1d_8_input"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЃB 
J__inference_sequential_21_layer_call_and_return_conditional_losses_1930809conv1d_8_input"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
гBа
%__inference_signature_wrapper_1930846conv1d_8_input"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
оBл
*__inference_conv1d_8_layer_call_fn_1931937inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
E__inference_conv1d_8_layer_call_and_return_conditional_losses_1931953inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
пBм
+__inference_flatten_9_layer_call_fn_1931958inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
њBї
F__inference_flatten_9_layer_call_and_return_conditional_losses_1931964inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
хBт
1__inference_repeat_vector_2_layer_call_fn_1931969inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B§
L__inference_repeat_vector_2_layer_call_and_return_conditional_losses_1931977inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
,0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
.__inference_simple_rnn_6_layer_call_fn_1931988inputs/0"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
.__inference_simple_rnn_6_layer_call_fn_1931999inputs/0"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
.__inference_simple_rnn_6_layer_call_fn_1932010inputs"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
.__inference_simple_rnn_6_layer_call_fn_1932021inputs"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
БBЎ
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_1932129inputs/0"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
БBЎ
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_1932237inputs/0"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЏBЌ
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_1932345inputs"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЏBЌ
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_1932453inputs"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
5
G0
H1
I2"
trackable_list_wrapper
5
G0
H1
I2"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Џnon_trainable_variables
Аlayers
Бmetrics
 Вlayer_regularization_losses
Гlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
х
Дtrace_0
Еtrace_12Њ
3__inference_simple_rnn_cell_6_layer_call_fn_1934061
3__inference_simple_rnn_cell_6_layer_call_fn_1934075Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zДtrace_0zЕtrace_1

Жtrace_0
Зtrace_12р
N__inference_simple_rnn_cell_6_layer_call_and_return_conditional_losses_1934092
N__inference_simple_rnn_cell_6_layer_call_and_return_conditional_losses_1934109Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЖtrace_0zЗtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
50"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
'__inference_gru_9_layer_call_fn_1932464inputs/0"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
'__inference_gru_9_layer_call_fn_1932475inputs/0"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
'__inference_gru_9_layer_call_fn_1932486inputs"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
'__inference_gru_9_layer_call_fn_1932497inputs"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЊBЇ
B__inference_gru_9_layer_call_and_return_conditional_losses_1932875inputs/0"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЊBЇ
B__inference_gru_9_layer_call_and_return_conditional_losses_1933253inputs/0"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЈBЅ
B__inference_gru_9_layer_call_and_return_conditional_losses_1933631inputs"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЈBЅ
B__inference_gru_9_layer_call_and_return_conditional_losses_1934009inputs"д
ЫВЧ
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
5
J0
K1
L2"
trackable_list_wrapper
5
J0
K1
L2"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Иnon_trainable_variables
Йlayers
Кmetrics
 Лlayer_regularization_losses
Мlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
У2РН
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
У2РН
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
оBл
*__inference_dense_28_layer_call_fn_1934018inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
E__inference_dense_28_layer_call_and_return_conditional_losses_1934028inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
оBл
*__inference_dense_29_layer_call_fn_1934037inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
E__inference_dense_29_layer_call_and_return_conditional_losses_1934047inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
R
Н	variables
О	keras_api

Пtotal

Рcount"
_tf_keras_metric
R
С	variables
Т	keras_api

Уtotal

Фcount"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
3__inference_simple_rnn_cell_6_layer_call_fn_1934061inputsstates/0"Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
3__inference_simple_rnn_cell_6_layer_call_fn_1934075inputsstates/0"Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЇBЄ
N__inference_simple_rnn_cell_6_layer_call_and_return_conditional_losses_1934092inputsstates/0"Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЇBЄ
N__inference_simple_rnn_cell_6_layer_call_and_return_conditional_losses_1934109inputsstates/0"Н
ДВА
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
П0
Р1"
trackable_list_wrapper
.
Н	variables"
_generic_user_object
:  (2total
:  (2count
0
У0
Ф1"
trackable_list_wrapper
.
С	variables"
_generic_user_object
:  (2total
:  (2count
*:( 2Adam/conv1d_8/kernel/m
 : 2Adam/conv1d_8/bias/m
&:$ 2Adam/dense_28/kernel/m
 :2Adam/dense_28/bias/m
&:$2Adam/dense_29/kernel/m
 :2Adam/dense_29/bias/m
=:;	р@2,Adam/simple_rnn_6/simple_rnn_cell_6/kernel/m
F:D@@26Adam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/m
6:4@2*Adam/simple_rnn_6/simple_rnn_cell_6/bias/m
.:,@`2Adam/gru_9/gru_cell_9/kernel/m
8:6 `2(Adam/gru_9/gru_cell_9/recurrent_kernel/m
,:*`2Adam/gru_9/gru_cell_9/bias/m
*:( 2Adam/conv1d_8/kernel/v
 : 2Adam/conv1d_8/bias/v
&:$ 2Adam/dense_28/kernel/v
 :2Adam/dense_28/bias/v
&:$2Adam/dense_29/kernel/v
 :2Adam/dense_29/bias/v
=:;	р@2,Adam/simple_rnn_6/simple_rnn_cell_6/kernel/v
F:D@@26Adam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/v
6:4@2*Adam/simple_rnn_6/simple_rnn_cell_6/bias/v
.:,@`2Adam/gru_9/gru_cell_9/kernel/v
8:6 `2(Adam/gru_9/gru_cell_9/recurrent_kernel/v
,:*`2Adam/gru_9/gru_cell_9/bias/vЇ
"__inference__wrapped_model_1928366GIHJKL=>EF;Ђ8
1Ђ.
,)
conv1d_8_inputџџџџџџџџџ
Њ "3Њ0
.
dense_29"
dense_29џџџџџџџџџ­
E__inference_conv1d_8_layer_call_and_return_conditional_losses_1931953d3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ ")Ђ&

0џџџџџџџџџ 
 
*__inference_conv1d_8_layer_call_fn_1931937W3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ "џџџџџџџџџ Ѕ
E__inference_dense_28_layer_call_and_return_conditional_losses_1934028\=>/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "%Ђ"

0џџџџџџџџџ
 }
*__inference_dense_28_layer_call_fn_1934018O=>/Ђ,
%Ђ"
 
inputsџџџџџџџџџ 
Њ "џџџџџџџџџЅ
E__inference_dense_29_layer_call_and_return_conditional_losses_1934047\EF/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 }
*__inference_dense_29_layer_call_fn_1934037OEF/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџЇ
F__inference_flatten_9_layer_call_and_return_conditional_losses_1931964]3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ 
Њ "&Ђ#

0џџџџџџџџџр
 
+__inference_flatten_9_layer_call_fn_1931958P3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ 
Њ "џџџџџџџџџрУ
B__inference_gru_9_layer_call_and_return_conditional_losses_1932875}JKLOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ@

 
p 

 
Њ "%Ђ"

0џџџџџџџџџ 
 У
B__inference_gru_9_layer_call_and_return_conditional_losses_1933253}JKLOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ@

 
p

 
Њ "%Ђ"

0џџџџџџџџџ 
 Г
B__inference_gru_9_layer_call_and_return_conditional_losses_1933631mJKL?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ@

 
p 

 
Њ "%Ђ"

0џџџџџџџџџ 
 Г
B__inference_gru_9_layer_call_and_return_conditional_losses_1934009mJKL?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ@

 
p

 
Њ "%Ђ"

0џџџџџџџџџ 
 
'__inference_gru_9_layer_call_fn_1932464pJKLOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ@

 
p 

 
Њ "џџџџџџџџџ 
'__inference_gru_9_layer_call_fn_1932475pJKLOЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ@

 
p

 
Њ "џџџџџџџџџ 
'__inference_gru_9_layer_call_fn_1932486`JKL?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ@

 
p 

 
Њ "џџџџџџџџџ 
'__inference_gru_9_layer_call_fn_1932497`JKL?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ@

 
p

 
Њ "џџџџџџџџџ О
L__inference_repeat_vector_2_layer_call_and_return_conditional_losses_1931977n8Ђ5
.Ђ+
)&
inputsџџџџџџџџџџџџџџџџџџ
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџ
 
1__inference_repeat_vector_2_layer_call_fn_1931969a8Ђ5
.Ђ+
)&
inputsџџџџџџџџџџџџџџџџџџ
Њ "%"џџџџџџџџџџџџџџџџџџШ
J__inference_sequential_21_layer_call_and_return_conditional_losses_1930774zGIHJKL=>EFCЂ@
9Ђ6
,)
conv1d_8_inputџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Ш
J__inference_sequential_21_layer_call_and_return_conditional_losses_1930809zGIHJKL=>EFCЂ@
9Ђ6
,)
conv1d_8_inputџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 Р
J__inference_sequential_21_layer_call_and_return_conditional_losses_1931416rGIHJKL=>EF;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 Р
J__inference_sequential_21_layer_call_and_return_conditional_losses_1931928rGIHJKL=>EF;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
  
/__inference_sequential_21_layer_call_fn_1930051mGIHJKL=>EFCЂ@
9Ђ6
,)
conv1d_8_inputџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ 
/__inference_sequential_21_layer_call_fn_1930739mGIHJKL=>EFCЂ@
9Ђ6
,)
conv1d_8_inputџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
/__inference_sequential_21_layer_call_fn_1930875eGIHJKL=>EF;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
/__inference_sequential_21_layer_call_fn_1930904eGIHJKL=>EF;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџМ
%__inference_signature_wrapper_1930846GIHJKL=>EFMЂJ
Ђ 
CЊ@
>
conv1d_8_input,)
conv1d_8_inputџџџџџџџџџ"3Њ0
.
dense_29"
dense_29џџџџџџџџџй
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_1932129GIHPЂM
FЂC
52
0-
inputs/0џџџџџџџџџџџџџџџџџџр

 
p 

 
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџ@
 й
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_1932237GIHPЂM
FЂC
52
0-
inputs/0џџџџџџџџџџџџџџџџџџр

 
p

 
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџ@
 П
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_1932345rGIH@Ђ=
6Ђ3
%"
inputsџџџџџџџџџр

 
p 

 
Њ ")Ђ&

0џџџџџџџџџ@
 П
I__inference_simple_rnn_6_layer_call_and_return_conditional_losses_1932453rGIH@Ђ=
6Ђ3
%"
inputsџџџџџџџџџр

 
p

 
Њ ")Ђ&

0џџџџџџџџџ@
 А
.__inference_simple_rnn_6_layer_call_fn_1931988~GIHPЂM
FЂC
52
0-
inputs/0џџџџџџџџџџџџџџџџџџр

 
p 

 
Њ "%"џџџџџџџџџџџџџџџџџџ@А
.__inference_simple_rnn_6_layer_call_fn_1931999~GIHPЂM
FЂC
52
0-
inputs/0џџџџџџџџџџџџџџџџџџр

 
p

 
Њ "%"џџџџџџџџџџџџџџџџџџ@
.__inference_simple_rnn_6_layer_call_fn_1932010eGIH@Ђ=
6Ђ3
%"
inputsџџџџџџџџџр

 
p 

 
Њ "џџџџџџџџџ@
.__inference_simple_rnn_6_layer_call_fn_1932021eGIH@Ђ=
6Ђ3
%"
inputsџџџџџџџџџр

 
p

 
Њ "џџџџџџџџџ@
N__inference_simple_rnn_cell_6_layer_call_and_return_conditional_losses_1934092ИGIH]ЂZ
SЂP
!
inputsџџџџџџџџџр
'Ђ$
"
states/0џџџџџџџџџ@
p 
Њ "RЂO
HЂE

0/0џџџџџџџџџ@
$!

0/1/0џџџџџџџџџ@
 
N__inference_simple_rnn_cell_6_layer_call_and_return_conditional_losses_1934109ИGIH]ЂZ
SЂP
!
inputsџџџџџџџџџр
'Ђ$
"
states/0џџџџџџџџџ@
p
Њ "RЂO
HЂE

0/0џџџџџџџџџ@
$!

0/1/0џџџџџџџџџ@
 т
3__inference_simple_rnn_cell_6_layer_call_fn_1934061ЊGIH]ЂZ
SЂP
!
inputsџџџџџџџџџр
'Ђ$
"
states/0џџџџџџџџџ@
p 
Њ "DЂA

0џџџџџџџџџ@
"

1/0џџџџџџџџџ@т
3__inference_simple_rnn_cell_6_layer_call_fn_1934075ЊGIH]ЂZ
SЂP
!
inputsџџџџџџџџџр
'Ђ$
"
states/0џџџџџџџџџ@
p
Њ "DЂA

0џџџџџџџџџ@
"

1/0џџџџџџџџџ@