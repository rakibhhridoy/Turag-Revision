Ог$
ѕХ
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
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
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
Ў
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
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

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
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
"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8и"
Ќ
*Adam/simple_rnn_6/simple_rnn_cell_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/simple_rnn_6/simple_rnn_cell_6/bias/v
Ѕ
>Adam/simple_rnn_6/simple_rnn_cell_6/bias/v/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_6/simple_rnn_cell_6/bias/v*
_output_shapes
:*
dtype0
Ш
6Adam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*G
shared_name86Adam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/v
С
JAdam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/v*
_output_shapes

:*
dtype0
Д
,Adam/simple_rnn_6/simple_rnn_cell_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*=
shared_name.,Adam/simple_rnn_6/simple_rnn_cell_6/kernel/v
­
@Adam/simple_rnn_6/simple_rnn_cell_6/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_6/simple_rnn_cell_6/kernel/v*
_output_shapes

:
*
dtype0
Ќ
*Adam/simple_rnn_5/simple_rnn_cell_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*;
shared_name,*Adam/simple_rnn_5/simple_rnn_cell_5/bias/v
Ѕ
>Adam/simple_rnn_5/simple_rnn_cell_5/bias/v/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_5/simple_rnn_cell_5/bias/v*
_output_shapes
:
*
dtype0
Ш
6Adam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*G
shared_name86Adam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/v
С
JAdam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/v*
_output_shapes

:

*
dtype0
Д
,Adam/simple_rnn_5/simple_rnn_cell_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*=
shared_name.,Adam/simple_rnn_5/simple_rnn_cell_5/kernel/v
­
@Adam/simple_rnn_5/simple_rnn_cell_5/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_5/simple_rnn_cell_5/kernel/v*
_output_shapes

:
*
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:*
dtype0

Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes

:*
dtype0
Ќ
*Adam/simple_rnn_6/simple_rnn_cell_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/simple_rnn_6/simple_rnn_cell_6/bias/m
Ѕ
>Adam/simple_rnn_6/simple_rnn_cell_6/bias/m/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_6/simple_rnn_cell_6/bias/m*
_output_shapes
:*
dtype0
Ш
6Adam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*G
shared_name86Adam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/m
С
JAdam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/m*
_output_shapes

:*
dtype0
Д
,Adam/simple_rnn_6/simple_rnn_cell_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*=
shared_name.,Adam/simple_rnn_6/simple_rnn_cell_6/kernel/m
­
@Adam/simple_rnn_6/simple_rnn_cell_6/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_6/simple_rnn_cell_6/kernel/m*
_output_shapes

:
*
dtype0
Ќ
*Adam/simple_rnn_5/simple_rnn_cell_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*;
shared_name,*Adam/simple_rnn_5/simple_rnn_cell_5/bias/m
Ѕ
>Adam/simple_rnn_5/simple_rnn_cell_5/bias/m/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_5/simple_rnn_cell_5/bias/m*
_output_shapes
:
*
dtype0
Ш
6Adam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*G
shared_name86Adam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/m
С
JAdam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/m*
_output_shapes

:

*
dtype0
Д
,Adam/simple_rnn_5/simple_rnn_cell_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*=
shared_name.,Adam/simple_rnn_5/simple_rnn_cell_5/kernel/m
­
@Adam/simple_rnn_5/simple_rnn_cell_5/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_5/simple_rnn_cell_5/kernel/m*
_output_shapes

:
*
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:*
dtype0

Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes

:*
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

#simple_rnn_6/simple_rnn_cell_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#simple_rnn_6/simple_rnn_cell_6/bias

7simple_rnn_6/simple_rnn_cell_6/bias/Read/ReadVariableOpReadVariableOp#simple_rnn_6/simple_rnn_cell_6/bias*
_output_shapes
:*
dtype0
К
/simple_rnn_6/simple_rnn_cell_6/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*@
shared_name1/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel
Г
Csimple_rnn_6/simple_rnn_cell_6/recurrent_kernel/Read/ReadVariableOpReadVariableOp/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel*
_output_shapes

:*
dtype0
І
%simple_rnn_6/simple_rnn_cell_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*6
shared_name'%simple_rnn_6/simple_rnn_cell_6/kernel

9simple_rnn_6/simple_rnn_cell_6/kernel/Read/ReadVariableOpReadVariableOp%simple_rnn_6/simple_rnn_cell_6/kernel*
_output_shapes

:
*
dtype0

#simple_rnn_5/simple_rnn_cell_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*4
shared_name%#simple_rnn_5/simple_rnn_cell_5/bias

7simple_rnn_5/simple_rnn_cell_5/bias/Read/ReadVariableOpReadVariableOp#simple_rnn_5/simple_rnn_cell_5/bias*
_output_shapes
:
*
dtype0
К
/simple_rnn_5/simple_rnn_cell_5/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*@
shared_name1/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel
Г
Csimple_rnn_5/simple_rnn_cell_5/recurrent_kernel/Read/ReadVariableOpReadVariableOp/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel*
_output_shapes

:

*
dtype0
І
%simple_rnn_5/simple_rnn_cell_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*6
shared_name'%simple_rnn_5/simple_rnn_cell_5/kernel

9simple_rnn_5/simple_rnn_cell_5/kernel/Read/ReadVariableOpReadVariableOp%simple_rnn_5/simple_rnn_cell_5/kernel*
_output_shapes

:
*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:*
dtype0

serving_default_model_2_inputPlaceholder*+
_output_shapes
:џџџџџџџџџ*
dtype0* 
shape:џџџџџџџџџ
ф
StatefulPartitionedCallStatefulPartitionedCallserving_default_model_2_input%simple_rnn_5/simple_rnn_cell_5/kernel#simple_rnn_5/simple_rnn_cell_5/bias/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel%simple_rnn_6/simple_rnn_cell_6/kernel#simple_rnn_6/simple_rnn_cell_6/bias/simple_rnn_6/simple_rnn_cell_6/recurrent_kerneldense_2/kerneldense_2/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_240023

NoOpNoOp
ОM
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*љL
valueяLBьL BхL

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	_default_save_signature

	optimizer

signatures*
ј
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
І
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
<
0
1
 2
!3
"4
#5
6
7*
<
0
1
 2
!3
"4
#5
6
7*
* 
А
$non_trainable_variables

%layers
&metrics
'layer_regularization_losses
(layer_metrics
	variables
trainable_variables
regularization_losses
__call__
	_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
)trace_0
*trace_1
+trace_2
,trace_3* 
6
-trace_0
.trace_1
/trace_2
0trace_3* 
* 
ф
1iter

2beta_1

3beta_2
	4decay
5learning_ratemЎmЏmАmБ mВ!mГ"mД#mЕvЖvЗvИvЙ vК!vЛ"vМ#vН*

6serving_default* 
* 
Њ
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses
=cell
>
state_spec*

?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses* 
Њ
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses
Kcell
L
state_spec*
.
0
1
 2
!3
"4
#5*
.
0
1
 2
!3
"4
#5*
* 

Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
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

0
1*

0
1*
* 

Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

_trace_0* 

`trace_0* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%simple_rnn_5/simple_rnn_cell_5/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#simple_rnn_5/simple_rnn_cell_5/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%simple_rnn_6/simple_rnn_cell_6/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#simple_rnn_6/simple_rnn_cell_6/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

a0
b1*
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

0
1
 2*

0
1
 2*
* 


cstates
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses*
6
itrace_0
jtrace_1
ktrace_2
ltrace_3* 
6
mtrace_0
ntrace_1
otrace_2
ptrace_3* 
г
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses
w_random_generator

kernel
recurrent_kernel
 bias*
* 
* 
* 
* 

xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses* 

}trace_0* 

~trace_0* 

!0
"1
#2*

!0
"1
#2*
* 
Є

states
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses*
:
trace_0
trace_1
trace_2
trace_3* 
:
trace_0
trace_1
trace_2
trace_3* 
к
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator

!kernel
"recurrent_kernel
#bias*
* 
* 
 
0
1
2
3*
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
<
	variables
	keras_api

total

count*
<
	variables
	keras_api

total

count*
* 
* 

=0*
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
0
1
 2*

0
1
 2*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses*

Ёtrace_0
Ђtrace_1* 

Ѓtrace_0
Єtrace_1* 
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

K0*
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
!0
"1
#2*

!0
"1
#2*
* 

Ѕnon_trainable_variables
Іlayers
Їmetrics
 Јlayer_regularization_losses
Љlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

Њtrace_0
Ћtrace_1* 

Ќtrace_0
­trace_1* 
* 

0
1*

	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
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
{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/simple_rnn_5/simple_rnn_cell_5/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE6Adam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/simple_rnn_5/simple_rnn_cell_5/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/simple_rnn_6/simple_rnn_cell_6/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE6Adam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/simple_rnn_6/simple_rnn_cell_6/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/simple_rnn_5/simple_rnn_cell_5/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE6Adam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/simple_rnn_5/simple_rnn_cell_5/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/simple_rnn_6/simple_rnn_cell_6/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE6Adam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/simple_rnn_6/simple_rnn_cell_6/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
э
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp9simple_rnn_5/simple_rnn_cell_5/kernel/Read/ReadVariableOpCsimple_rnn_5/simple_rnn_cell_5/recurrent_kernel/Read/ReadVariableOp7simple_rnn_5/simple_rnn_cell_5/bias/Read/ReadVariableOp9simple_rnn_6/simple_rnn_cell_6/kernel/Read/ReadVariableOpCsimple_rnn_6/simple_rnn_cell_6/recurrent_kernel/Read/ReadVariableOp7simple_rnn_6/simple_rnn_cell_6/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp@Adam/simple_rnn_5/simple_rnn_cell_5/kernel/m/Read/ReadVariableOpJAdam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/m/Read/ReadVariableOp>Adam/simple_rnn_5/simple_rnn_cell_5/bias/m/Read/ReadVariableOp@Adam/simple_rnn_6/simple_rnn_cell_6/kernel/m/Read/ReadVariableOpJAdam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/m/Read/ReadVariableOp>Adam/simple_rnn_6/simple_rnn_cell_6/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp@Adam/simple_rnn_5/simple_rnn_cell_5/kernel/v/Read/ReadVariableOpJAdam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/v/Read/ReadVariableOp>Adam/simple_rnn_5/simple_rnn_cell_5/bias/v/Read/ReadVariableOp@Adam/simple_rnn_6/simple_rnn_cell_6/kernel/v/Read/ReadVariableOpJAdam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/v/Read/ReadVariableOp>Adam/simple_rnn_6/simple_rnn_cell_6/bias/v/Read/ReadVariableOpConst*.
Tin'
%2#	*
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
GPU 2J 8 *(
f#R!
__inference__traced_save_242281
д

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_2/kerneldense_2/bias%simple_rnn_5/simple_rnn_cell_5/kernel/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel#simple_rnn_5/simple_rnn_cell_5/bias%simple_rnn_6/simple_rnn_cell_6/kernel/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel#simple_rnn_6/simple_rnn_cell_6/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/dense_2/kernel/mAdam/dense_2/bias/m,Adam/simple_rnn_5/simple_rnn_cell_5/kernel/m6Adam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/m*Adam/simple_rnn_5/simple_rnn_cell_5/bias/m,Adam/simple_rnn_6/simple_rnn_cell_6/kernel/m6Adam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/m*Adam/simple_rnn_6/simple_rnn_cell_6/bias/mAdam/dense_2/kernel/vAdam/dense_2/bias/v,Adam/simple_rnn_5/simple_rnn_cell_5/kernel/v6Adam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/v*Adam/simple_rnn_5/simple_rnn_cell_5/bias/v,Adam/simple_rnn_6/simple_rnn_cell_6/kernel/v6Adam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/v*Adam/simple_rnn_6/simple_rnn_cell_6/bias/v*-
Tin&
$2"*
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
GPU 2J 8 *+
f&R$
"__inference__traced_restore_242390хв 
к
Њ
while_cond_241258
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_241258___redundant_placeholder04
0while_while_cond_241258___redundant_placeholder14
0while_while_cond_241258___redundant_placeholder24
0while_while_cond_241258___redundant_placeholder3
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
-: : : : :џџџџџџџџџ
: ::::: 
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
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
:

Ж
H__inference_sequential_5_layer_call_and_return_conditional_losses_239972
model_2_input 
model_2_239953:

model_2_239955:
 
model_2_239957:

 
model_2_239959:

model_2_239961: 
model_2_239963: 
dense_2_239966:
dense_2_239968:
identityЂdense_2/StatefulPartitionedCallЂmodel_2/StatefulPartitionedCallП
model_2/StatefulPartitionedCallStatefulPartitionedCallmodel_2_inputmodel_2_239953model_2_239955model_2_239957model_2_239959model_2_239961model_2_239963*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_239394
dense_2/StatefulPartitionedCallStatefulPartitionedCall(model_2/StatefulPartitionedCall:output:0dense_2_239966dense_2_239968*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_239829{
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ
NoOpNoOp ^dense_2/StatefulPartitionedCall ^model_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
model_2/StatefulPartitionedCallmodel_2/StatefulPartitionedCall:Z V
+
_output_shapes
:џџџџџџџџџ
'
_user_specified_namemodel_2_input
ю
Џ
H__inference_sequential_5_layer_call_and_return_conditional_losses_239910

inputs 
model_2_239891:

model_2_239893:
 
model_2_239895:

 
model_2_239897:

model_2_239899: 
model_2_239901: 
dense_2_239904:
dense_2_239906:
identityЂdense_2/StatefulPartitionedCallЂmodel_2/StatefulPartitionedCallИ
model_2/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_2_239891model_2_239893model_2_239895model_2_239897model_2_239899model_2_239901*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_239709
dense_2/StatefulPartitionedCallStatefulPartitionedCall(model_2/StatefulPartitionedCall:output:0dense_2_239904dense_2_239906*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_239829{
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ
NoOpNoOp ^dense_2/StatefulPartitionedCall ^model_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
model_2/StatefulPartitionedCallmodel_2/StatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
рM
Ш
3sequential_5_model_2_simple_rnn_5_while_body_238349`
\sequential_5_model_2_simple_rnn_5_while_sequential_5_model_2_simple_rnn_5_while_loop_counterf
bsequential_5_model_2_simple_rnn_5_while_sequential_5_model_2_simple_rnn_5_while_maximum_iterations7
3sequential_5_model_2_simple_rnn_5_while_placeholder9
5sequential_5_model_2_simple_rnn_5_while_placeholder_19
5sequential_5_model_2_simple_rnn_5_while_placeholder_2_
[sequential_5_model_2_simple_rnn_5_while_sequential_5_model_2_simple_rnn_5_strided_slice_1_0
sequential_5_model_2_simple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_sequential_5_model_2_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0l
Zsequential_5_model_2_simple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resource_0:
i
[sequential_5_model_2_simple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resource_0:
n
\sequential_5_model_2_simple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0:

4
0sequential_5_model_2_simple_rnn_5_while_identity6
2sequential_5_model_2_simple_rnn_5_while_identity_16
2sequential_5_model_2_simple_rnn_5_while_identity_26
2sequential_5_model_2_simple_rnn_5_while_identity_36
2sequential_5_model_2_simple_rnn_5_while_identity_4]
Ysequential_5_model_2_simple_rnn_5_while_sequential_5_model_2_simple_rnn_5_strided_slice_1
sequential_5_model_2_simple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_sequential_5_model_2_simple_rnn_5_tensorarrayunstack_tensorlistfromtensorj
Xsequential_5_model_2_simple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resource:
g
Ysequential_5_model_2_simple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resource:
l
Zsequential_5_model_2_simple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource:

ЂPsequential_5/model_2/simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOpЂOsequential_5/model_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOpЂQsequential_5/model_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpЊ
Ysequential_5/model_2/simple_rnn_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   б
Ksequential_5/model_2/simple_rnn_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_5_model_2_simple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_sequential_5_model_2_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_03sequential_5_model_2_simple_rnn_5_while_placeholderbsequential_5/model_2/simple_rnn_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0ъ
Osequential_5/model_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOpZsequential_5_model_2_simple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype0Љ
@sequential_5/model_2/simple_rnn_5/while/simple_rnn_cell_5/MatMulMatMulRsequential_5/model_2/simple_rnn_5/while/TensorArrayV2Read/TensorListGetItem:item:0Wsequential_5/model_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
ш
Psequential_5/model_2/simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp[sequential_5_model_2_simple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resource_0*
_output_shapes
:
*
dtype0Є
Asequential_5/model_2/simple_rnn_5/while/simple_rnn_cell_5/BiasAddBiasAddJsequential_5/model_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul:product:0Xsequential_5/model_2/simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
ю
Qsequential_5/model_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp\sequential_5_model_2_simple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:

*
dtype0
Bsequential_5/model_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1MatMul5sequential_5_model_2_simple_rnn_5_while_placeholder_2Ysequential_5/model_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ

=sequential_5/model_2/simple_rnn_5/while/simple_rnn_cell_5/addAddV2Jsequential_5/model_2/simple_rnn_5/while/simple_rnn_cell_5/BiasAdd:output:0Lsequential_5/model_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ
Л
>sequential_5/model_2/simple_rnn_5/while/simple_rnn_cell_5/ReluReluAsequential_5/model_2/simple_rnn_5/while/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ

Rsequential_5/model_2/simple_rnn_5/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
Lsequential_5/model_2/simple_rnn_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem5sequential_5_model_2_simple_rnn_5_while_placeholder_1[sequential_5/model_2/simple_rnn_5/while/TensorArrayV2Write/TensorListSetItem/index:output:0Lsequential_5/model_2/simple_rnn_5/while/simple_rnn_cell_5/Relu:activations:0*
_output_shapes
: *
element_dtype0:щшвo
-sequential_5/model_2/simple_rnn_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Т
+sequential_5/model_2/simple_rnn_5/while/addAddV23sequential_5_model_2_simple_rnn_5_while_placeholder6sequential_5/model_2/simple_rnn_5/while/add/y:output:0*
T0*
_output_shapes
: q
/sequential_5/model_2/simple_rnn_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :я
-sequential_5/model_2/simple_rnn_5/while/add_1AddV2\sequential_5_model_2_simple_rnn_5_while_sequential_5_model_2_simple_rnn_5_while_loop_counter8sequential_5/model_2/simple_rnn_5/while/add_1/y:output:0*
T0*
_output_shapes
: П
0sequential_5/model_2/simple_rnn_5/while/IdentityIdentity1sequential_5/model_2/simple_rnn_5/while/add_1:z:0-^sequential_5/model_2/simple_rnn_5/while/NoOp*
T0*
_output_shapes
: ђ
2sequential_5/model_2/simple_rnn_5/while/Identity_1Identitybsequential_5_model_2_simple_rnn_5_while_sequential_5_model_2_simple_rnn_5_while_maximum_iterations-^sequential_5/model_2/simple_rnn_5/while/NoOp*
T0*
_output_shapes
: П
2sequential_5/model_2/simple_rnn_5/while/Identity_2Identity/sequential_5/model_2/simple_rnn_5/while/add:z:0-^sequential_5/model_2/simple_rnn_5/while/NoOp*
T0*
_output_shapes
: ь
2sequential_5/model_2/simple_rnn_5/while/Identity_3Identity\sequential_5/model_2/simple_rnn_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:0-^sequential_5/model_2/simple_rnn_5/while/NoOp*
T0*
_output_shapes
: э
2sequential_5/model_2/simple_rnn_5/while/Identity_4IdentityLsequential_5/model_2/simple_rnn_5/while/simple_rnn_cell_5/Relu:activations:0-^sequential_5/model_2/simple_rnn_5/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
ч
,sequential_5/model_2/simple_rnn_5/while/NoOpNoOpQ^sequential_5/model_2/simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOpP^sequential_5/model_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOpR^sequential_5/model_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "m
0sequential_5_model_2_simple_rnn_5_while_identity9sequential_5/model_2/simple_rnn_5/while/Identity:output:0"q
2sequential_5_model_2_simple_rnn_5_while_identity_1;sequential_5/model_2/simple_rnn_5/while/Identity_1:output:0"q
2sequential_5_model_2_simple_rnn_5_while_identity_2;sequential_5/model_2/simple_rnn_5/while/Identity_2:output:0"q
2sequential_5_model_2_simple_rnn_5_while_identity_3;sequential_5/model_2/simple_rnn_5/while/Identity_3:output:0"q
2sequential_5_model_2_simple_rnn_5_while_identity_4;sequential_5/model_2/simple_rnn_5/while/Identity_4:output:0"И
Ysequential_5_model_2_simple_rnn_5_while_sequential_5_model_2_simple_rnn_5_strided_slice_1[sequential_5_model_2_simple_rnn_5_while_sequential_5_model_2_simple_rnn_5_strided_slice_1_0"И
Ysequential_5_model_2_simple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resource[sequential_5_model_2_simple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resource_0"К
Zsequential_5_model_2_simple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource\sequential_5_model_2_simple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0"Ж
Xsequential_5_model_2_simple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resourceZsequential_5_model_2_simple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resource_0"В
sequential_5_model_2_simple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_sequential_5_model_2_simple_rnn_5_tensorarrayunstack_tensorlistfromtensorsequential_5_model_2_simple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_sequential_5_model_2_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ
: : : : : 2Є
Psequential_5/model_2/simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOpPsequential_5/model_2/simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp2Ђ
Osequential_5/model_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOpOsequential_5/model_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp2І
Qsequential_5/model_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpQsequential_5/model_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
: 
П,
Ш
while_body_241861
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_6_matmul_readvariableop_resource_0:
G
9while_simple_rnn_cell_6_biasadd_readvariableop_resource_0:L
:while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_6_matmul_readvariableop_resource:
E
7while_simple_rnn_cell_6_biasadd_readvariableop_resource:J
8while_simple_rnn_cell_6_matmul_1_readvariableop_resource:Ђ.while/simple_rnn_cell_6/BiasAdd/ReadVariableOpЂ-while/simple_rnn_cell_6/MatMul/ReadVariableOpЂ/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ
*
element_dtype0І
-while/simple_rnn_cell_6/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_6_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype0У
while/simple_rnn_cell_6/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЄ
.while/simple_rnn_cell_6/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_6_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0О
while/simple_rnn_cell_6/BiasAddBiasAdd(while/simple_rnn_cell_6/MatMul:product:06while/simple_rnn_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЊ
/while/simple_rnn_cell_6/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0Њ
 while/simple_rnn_cell_6/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЌ
while/simple_rnn_cell_6/addAddV2(while/simple_rnn_cell_6/BiasAdd:output:0*while/simple_rnn_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџw
while/simple_rnn_cell_6/ReluReluwhile/simple_rnn_cell_6/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџг
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder*while/simple_rnn_cell_6/Relu:activations:0*
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
: 
while/Identity_4Identity*while/simple_rnn_cell_6/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџп

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
%: : : : :џџџџџџџџџ: : : : : 2`
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
П,
Ш
while_body_239464
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_6_matmul_readvariableop_resource_0:
G
9while_simple_rnn_cell_6_biasadd_readvariableop_resource_0:L
:while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_6_matmul_readvariableop_resource:
E
7while_simple_rnn_cell_6_biasadd_readvariableop_resource:J
8while_simple_rnn_cell_6_matmul_1_readvariableop_resource:Ђ.while/simple_rnn_cell_6/BiasAdd/ReadVariableOpЂ-while/simple_rnn_cell_6/MatMul/ReadVariableOpЂ/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ
*
element_dtype0І
-while/simple_rnn_cell_6/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_6_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype0У
while/simple_rnn_cell_6/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЄ
.while/simple_rnn_cell_6/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_6_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0О
while/simple_rnn_cell_6/BiasAddBiasAdd(while/simple_rnn_cell_6/MatMul:product:06while/simple_rnn_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЊ
/while/simple_rnn_cell_6/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0Њ
 while/simple_rnn_cell_6/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЌ
while/simple_rnn_cell_6/addAddV2(while/simple_rnn_cell_6/BiasAdd:output:0*while/simple_rnn_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџw
while/simple_rnn_cell_6/ReluReluwhile/simple_rnn_cell_6/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџг
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder*while/simple_rnn_cell_6/Relu:activations:0*
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
: 
while/Identity_4Identity*while/simple_rnn_cell_6/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџп

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
%: : : : :џџџџџџџџџ: : : : : 2`
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
Л
g
K__inference_repeat_vector_2_layer_call_and_return_conditional_losses_238858

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
valueB"         p
TileTileExpandDims:output:0stack:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџb
IdentityIdentityTile:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџџџџџџџџџџ:X T
0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Й

к
2__inference_simple_rnn_cell_6_layer_call_fn_242111

inputs
states_0
unknown:

	unknown_0:
	unknown_1:
identity

identity_1ЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_simple_rnn_cell_6_layer_call_and_return_conditional_losses_238909o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ
:џџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0
ТA
П
&model_2_simple_rnn_5_while_body_240108F
Bmodel_2_simple_rnn_5_while_model_2_simple_rnn_5_while_loop_counterL
Hmodel_2_simple_rnn_5_while_model_2_simple_rnn_5_while_maximum_iterations*
&model_2_simple_rnn_5_while_placeholder,
(model_2_simple_rnn_5_while_placeholder_1,
(model_2_simple_rnn_5_while_placeholder_2E
Amodel_2_simple_rnn_5_while_model_2_simple_rnn_5_strided_slice_1_0
}model_2_simple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_model_2_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0_
Mmodel_2_simple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resource_0:
\
Nmodel_2_simple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resource_0:
a
Omodel_2_simple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0:

'
#model_2_simple_rnn_5_while_identity)
%model_2_simple_rnn_5_while_identity_1)
%model_2_simple_rnn_5_while_identity_2)
%model_2_simple_rnn_5_while_identity_3)
%model_2_simple_rnn_5_while_identity_4C
?model_2_simple_rnn_5_while_model_2_simple_rnn_5_strided_slice_1
{model_2_simple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_model_2_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor]
Kmodel_2_simple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resource:
Z
Lmodel_2_simple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resource:
_
Mmodel_2_simple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource:

ЂCmodel_2/simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOpЂBmodel_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOpЂDmodel_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp
Lmodel_2/simple_rnn_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
>model_2/simple_rnn_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}model_2_simple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_model_2_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0&model_2_simple_rnn_5_while_placeholderUmodel_2/simple_rnn_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0а
Bmodel_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOpMmodel_2_simple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype0
3model_2/simple_rnn_5/while/simple_rnn_cell_5/MatMulMatMulEmodel_2/simple_rnn_5/while/TensorArrayV2Read/TensorListGetItem:item:0Jmodel_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Ю
Cmodel_2/simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOpNmodel_2_simple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resource_0*
_output_shapes
:
*
dtype0§
4model_2/simple_rnn_5/while/simple_rnn_cell_5/BiasAddBiasAdd=model_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul:product:0Kmodel_2/simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
д
Dmodel_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOpOmodel_2_simple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:

*
dtype0щ
5model_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1MatMul(model_2_simple_rnn_5_while_placeholder_2Lmodel_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
ы
0model_2/simple_rnn_5/while/simple_rnn_cell_5/addAddV2=model_2/simple_rnn_5/while/simple_rnn_cell_5/BiasAdd:output:0?model_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ
Ё
1model_2/simple_rnn_5/while/simple_rnn_cell_5/ReluRelu4model_2/simple_rnn_5/while/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ

Emodel_2/simple_rnn_5/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Я
?model_2/simple_rnn_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(model_2_simple_rnn_5_while_placeholder_1Nmodel_2/simple_rnn_5/while/TensorArrayV2Write/TensorListSetItem/index:output:0?model_2/simple_rnn_5/while/simple_rnn_cell_5/Relu:activations:0*
_output_shapes
: *
element_dtype0:щшвb
 model_2/simple_rnn_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
model_2/simple_rnn_5/while/addAddV2&model_2_simple_rnn_5_while_placeholder)model_2/simple_rnn_5/while/add/y:output:0*
T0*
_output_shapes
: d
"model_2/simple_rnn_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Л
 model_2/simple_rnn_5/while/add_1AddV2Bmodel_2_simple_rnn_5_while_model_2_simple_rnn_5_while_loop_counter+model_2/simple_rnn_5/while/add_1/y:output:0*
T0*
_output_shapes
: 
#model_2/simple_rnn_5/while/IdentityIdentity$model_2/simple_rnn_5/while/add_1:z:0 ^model_2/simple_rnn_5/while/NoOp*
T0*
_output_shapes
: О
%model_2/simple_rnn_5/while/Identity_1IdentityHmodel_2_simple_rnn_5_while_model_2_simple_rnn_5_while_maximum_iterations ^model_2/simple_rnn_5/while/NoOp*
T0*
_output_shapes
: 
%model_2/simple_rnn_5/while/Identity_2Identity"model_2/simple_rnn_5/while/add:z:0 ^model_2/simple_rnn_5/while/NoOp*
T0*
_output_shapes
: Х
%model_2/simple_rnn_5/while/Identity_3IdentityOmodel_2/simple_rnn_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^model_2/simple_rnn_5/while/NoOp*
T0*
_output_shapes
: Ц
%model_2/simple_rnn_5/while/Identity_4Identity?model_2/simple_rnn_5/while/simple_rnn_cell_5/Relu:activations:0 ^model_2/simple_rnn_5/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
Г
model_2/simple_rnn_5/while/NoOpNoOpD^model_2/simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOpC^model_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOpE^model_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "S
#model_2_simple_rnn_5_while_identity,model_2/simple_rnn_5/while/Identity:output:0"W
%model_2_simple_rnn_5_while_identity_1.model_2/simple_rnn_5/while/Identity_1:output:0"W
%model_2_simple_rnn_5_while_identity_2.model_2/simple_rnn_5/while/Identity_2:output:0"W
%model_2_simple_rnn_5_while_identity_3.model_2/simple_rnn_5/while/Identity_3:output:0"W
%model_2_simple_rnn_5_while_identity_4.model_2/simple_rnn_5/while/Identity_4:output:0"
?model_2_simple_rnn_5_while_model_2_simple_rnn_5_strided_slice_1Amodel_2_simple_rnn_5_while_model_2_simple_rnn_5_strided_slice_1_0"
Lmodel_2_simple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resourceNmodel_2_simple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resource_0" 
Mmodel_2_simple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resourceOmodel_2_simple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0"
Kmodel_2_simple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resourceMmodel_2_simple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resource_0"ќ
{model_2_simple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_model_2_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor}model_2_simple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_model_2_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ
: : : : : 2
Cmodel_2/simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOpCmodel_2/simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp2
Bmodel_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOpBmodel_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp2
Dmodel_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpDmodel_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
: 
к
Њ
while_cond_239080
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_239080___redundant_placeholder04
0while_while_cond_239080___redundant_placeholder14
0while_while_cond_239080___redundant_placeholder24
0while_while_cond_239080___redundant_placeholder3
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
-: : : : :џџџџџџџџџ: ::::: 
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
р=
Н
H__inference_simple_rnn_6_layer_call_and_return_conditional_losses_241819
inputs_0B
0simple_rnn_cell_6_matmul_readvariableop_resource:
?
1simple_rnn_cell_6_biasadd_readvariableop_resource:D
2simple_rnn_cell_6_matmul_1_readvariableop_resource:
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
value	B :s
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
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
D
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
valueB"џџџџ
   р
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ
*
shrink_axis_mask
'simple_rnn_cell_6/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_6_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
simple_rnn_cell_6/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
(simple_rnn_cell_6/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ќ
simple_rnn_cell_6/BiasAddBiasAdd"simple_rnn_cell_6/MatMul:product:00simple_rnn_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
)simple_rnn_cell_6/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_6_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0
simple_rnn_cell_6/MatMul_1MatMulzeros:output:01simple_rnn_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_6/addAddV2"simple_rnn_cell_6/BiasAdd:output:0$simple_rnn_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџk
simple_rnn_cell_6/ReluRelusimple_rnn_cell_6/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
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
value	B : и
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_6_matmul_readvariableop_resource1simple_rnn_cell_6_biasadd_readvariableop_resource2simple_rnn_cell_6_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_241753*
condR
while_cond_241752*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
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
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџk
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџЯ
NoOpNoOp)^simple_rnn_cell_6/BiasAdd/ReadVariableOp(^simple_rnn_cell_6/MatMul/ReadVariableOp*^simple_rnn_cell_6/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ
: : : 2T
(simple_rnn_cell_6/BiasAdd/ReadVariableOp(simple_rnn_cell_6/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_6/MatMul/ReadVariableOp'simple_rnn_cell_6/MatMul/ReadVariableOp2V
)simple_rnn_cell_6/MatMul_1/ReadVariableOp)simple_rnn_cell_6/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ

"
_user_specified_name
inputs/0
к
Њ
while_cond_238921
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_238921___redundant_placeholder04
0while_while_cond_238921___redundant_placeholder14
0while_while_cond_238921___redundant_placeholder24
0while_while_cond_238921___redundant_placeholder3
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
-: : : : :џџџџџџџџџ: ::::: 
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
А
Й
-__inference_simple_rnn_6_layer_call_fn_241581
inputs_0
unknown:

	unknown_0:
	unknown_1:
identityЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_simple_rnn_6_layer_call_and_return_conditional_losses_239144|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ

"
_user_specified_name
inputs/0
Ђ=
Л
H__inference_simple_rnn_6_layer_call_and_return_conditional_losses_242035

inputsB
0simple_rnn_cell_6_matmul_readvariableop_resource:
?
1simple_rnn_cell_6_biasadd_readvariableop_resource:D
2simple_rnn_cell_6_matmul_1_readvariableop_resource:
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
value	B :s
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
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
D
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
valueB"џџџџ
   р
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ
*
shrink_axis_mask
'simple_rnn_cell_6/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_6_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
simple_rnn_cell_6/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
(simple_rnn_cell_6/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ќ
simple_rnn_cell_6/BiasAddBiasAdd"simple_rnn_cell_6/MatMul:product:00simple_rnn_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
)simple_rnn_cell_6/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_6_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0
simple_rnn_cell_6/MatMul_1MatMulzeros:output:01simple_rnn_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_6/addAddV2"simple_rnn_cell_6/BiasAdd:output:0$simple_rnn_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџk
simple_rnn_cell_6/ReluRelusimple_rnn_cell_6/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
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
value	B : и
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_6_matmul_readvariableop_resource1simple_rnn_cell_6_biasadd_readvariableop_resource2simple_rnn_cell_6_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_241969*
condR
while_cond_241968*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Т
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
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
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџb
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџЯ
NoOpNoOp)^simple_rnn_cell_6/BiasAdd/ReadVariableOp(^simple_rnn_cell_6/MatMul/ReadVariableOp*^simple_rnn_cell_6/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ
: : : 2T
(simple_rnn_cell_6/BiasAdd/ReadVariableOp(simple_rnn_cell_6/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_6/MatMul/ReadVariableOp'simple_rnn_cell_6/MatMul/ReadVariableOp2V
)simple_rnn_cell_6/MatMul_1/ReadVariableOp)simple_rnn_cell_6/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameinputs
і 
б
while_body_238922
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
 while_simple_rnn_cell_6_238944_0:
.
 while_simple_rnn_cell_6_238946_0:2
 while_simple_rnn_cell_6_238948_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
while_simple_rnn_cell_6_238944:
,
while_simple_rnn_cell_6_238946:0
while_simple_rnn_cell_6_238948:Ђ/while/simple_rnn_cell_6/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ
*
element_dtype0Ђ
/while/simple_rnn_cell_6/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2 while_simple_rnn_cell_6_238944_0 while_simple_rnn_cell_6_238946_0 while_simple_rnn_cell_6_238948_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_simple_rnn_cell_6_layer_call_and_return_conditional_losses_238909с
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
:џџџџџџџџџ~

while/NoOpNoOp0^while/simple_rnn_cell_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"B
while_simple_rnn_cell_6_238944 while_simple_rnn_cell_6_238944_0"B
while_simple_rnn_cell_6_238946 while_simple_rnn_cell_6_238946_0"B
while_simple_rnn_cell_6_238948 while_simple_rnn_cell_6_238948_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ: : : : : 2b
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 

З
-__inference_simple_rnn_6_layer_call_fn_241603

inputs
unknown:

	unknown_0:
	unknown_1:
identityЂStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_simple_rnn_6_layer_call_and_return_conditional_losses_239530s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameinputs
Ђ=
Л
H__inference_simple_rnn_6_layer_call_and_return_conditional_losses_241927

inputsB
0simple_rnn_cell_6_matmul_readvariableop_resource:
?
1simple_rnn_cell_6_biasadd_readvariableop_resource:D
2simple_rnn_cell_6_matmul_1_readvariableop_resource:
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
value	B :s
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
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
D
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
valueB"џџџџ
   р
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ
*
shrink_axis_mask
'simple_rnn_cell_6/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_6_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
simple_rnn_cell_6/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
(simple_rnn_cell_6/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ќ
simple_rnn_cell_6/BiasAddBiasAdd"simple_rnn_cell_6/MatMul:product:00simple_rnn_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
)simple_rnn_cell_6/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_6_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0
simple_rnn_cell_6/MatMul_1MatMulzeros:output:01simple_rnn_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_6/addAddV2"simple_rnn_cell_6/BiasAdd:output:0$simple_rnn_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџk
simple_rnn_cell_6/ReluRelusimple_rnn_cell_6/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
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
value	B : и
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_6_matmul_readvariableop_resource1simple_rnn_cell_6_biasadd_readvariableop_resource2simple_rnn_cell_6_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_241861*
condR
while_cond_241860*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Т
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
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
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџb
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџЯ
NoOpNoOp)^simple_rnn_cell_6/BiasAdd/ReadVariableOp(^simple_rnn_cell_6/MatMul/ReadVariableOp*^simple_rnn_cell_6/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ
: : : 2T
(simple_rnn_cell_6/BiasAdd/ReadVariableOp(simple_rnn_cell_6/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_6/MatMul/ReadVariableOp'simple_rnn_cell_6/MatMul/ReadVariableOp2V
)simple_rnn_cell_6/MatMul_1/ReadVariableOp)simple_rnn_cell_6/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameinputs
ђ

C__inference_model_2_layer_call_and_return_conditional_losses_239760
input_6%
simple_rnn_5_239744:
!
simple_rnn_5_239746:
%
simple_rnn_5_239748:

%
simple_rnn_6_239752:
!
simple_rnn_6_239754:%
simple_rnn_6_239756:
identityЂ$simple_rnn_5/StatefulPartitionedCallЂ$simple_rnn_6/StatefulPartitionedCall
$simple_rnn_5/StatefulPartitionedCallStatefulPartitionedCallinput_6simple_rnn_5_239744simple_rnn_5_239746simple_rnn_5_239748*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_239269ё
repeat_vector_2/PartitionedCallPartitionedCall-simple_rnn_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_repeat_vector_2_layer_call_and_return_conditional_losses_238858Н
$simple_rnn_6/StatefulPartitionedCallStatefulPartitionedCall(repeat_vector_2/PartitionedCall:output:0simple_rnn_6_239752simple_rnn_6_239754simple_rnn_6_239756*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_simple_rnn_6_layer_call_and_return_conditional_losses_239385
IdentityIdentity-simple_rnn_6/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ
NoOpNoOp%^simple_rnn_5/StatefulPartitionedCall%^simple_rnn_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : : : : : 2L
$simple_rnn_5/StatefulPartitionedCall$simple_rnn_5/StatefulPartitionedCall2L
$simple_rnn_6/StatefulPartitionedCall$simple_rnn_6/StatefulPartitionedCall:T P
+
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_6
л-
Ш
while_body_241149
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_5_matmul_readvariableop_resource_0:
G
9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0:
L
:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0:


while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_5_matmul_readvariableop_resource:
E
7while_simple_rnn_cell_5_biasadd_readvariableop_resource:
J
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource:

Ђ.while/simple_rnn_cell_5/BiasAdd/ReadVariableOpЂ-while/simple_rnn_cell_5/MatMul/ReadVariableOpЂ/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0І
-while/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_5_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype0У
while/simple_rnn_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Є
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0*
_output_shapes
:
*
dtype0О
while/simple_rnn_cell_5/BiasAddBiasAdd(while/simple_rnn_cell_5/MatMul:product:06while/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Њ
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:

*
dtype0Њ
 while/simple_rnn_cell_5/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Ќ
while/simple_rnn_cell_5/addAddV2(while/simple_rnn_cell_5/BiasAdd:output:0*while/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ
w
while/simple_rnn_cell_5/ReluReluwhile/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ћ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0*while/simple_rnn_cell_5/Relu:activations:0*
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
: 
while/Identity_4Identity*while/simple_rnn_cell_5/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
п

while/NoOpNoOp/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_5_biasadd_readvariableop_resource9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_5_matmul_readvariableop_resource8while_simple_rnn_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ
: : : : : 2`
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_5/MatMul/ReadVariableOp-while/simple_rnn_cell_5/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
: 
П,
Ш
while_body_239319
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_6_matmul_readvariableop_resource_0:
G
9while_simple_rnn_cell_6_biasadd_readvariableop_resource_0:L
:while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_6_matmul_readvariableop_resource:
E
7while_simple_rnn_cell_6_biasadd_readvariableop_resource:J
8while_simple_rnn_cell_6_matmul_1_readvariableop_resource:Ђ.while/simple_rnn_cell_6/BiasAdd/ReadVariableOpЂ-while/simple_rnn_cell_6/MatMul/ReadVariableOpЂ/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ
*
element_dtype0І
-while/simple_rnn_cell_6/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_6_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype0У
while/simple_rnn_cell_6/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЄ
.while/simple_rnn_cell_6/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_6_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0О
while/simple_rnn_cell_6/BiasAddBiasAdd(while/simple_rnn_cell_6/MatMul:product:06while/simple_rnn_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЊ
/while/simple_rnn_cell_6/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0Њ
 while/simple_rnn_cell_6/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЌ
while/simple_rnn_cell_6/addAddV2(while/simple_rnn_cell_6/BiasAdd:output:0*while/simple_rnn_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџw
while/simple_rnn_cell_6/ReluReluwhile/simple_rnn_cell_6/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџг
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder*while/simple_rnn_cell_6/Relu:activations:0*
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
: 
while/Identity_4Identity*while/simple_rnn_cell_6/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџп

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
%: : : : :џџџџџџџџџ: : : : : 2`
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
Й

к
2__inference_simple_rnn_cell_5_layer_call_fn_242049

inputs
states_0
unknown:

	unknown_0:

	unknown_1:


identity

identity_1ЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ
:џџџџџџџџџ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_238598o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ:џџџџџџџџџ
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ

"
_user_specified_name
states/0
Ј>
Л
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_239662

inputsB
0simple_rnn_cell_5_matmul_readvariableop_resource:
?
1simple_rnn_cell_5_biasadd_readvariableop_resource:
D
2simple_rnn_cell_5_matmul_1_readvariableop_resource:


identityЂ(simple_rnn_cell_5/BiasAdd/ReadVariableOpЂ'simple_rnn_cell_5/MatMul/ReadVariableOpЂ)simple_rnn_cell_5/MatMul_1/ReadVariableOpЂwhile;
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
value	B :
s
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
:џџџџџџџџџ
c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџD
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
valueB"џџџџ   р
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
'simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_5_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
simple_rnn_cell_5/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ

(simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_5_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Ќ
simple_rnn_cell_5/BiasAddBiasAdd"simple_rnn_cell_5/MatMul:product:00simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ

)simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype0
simple_rnn_cell_5/MatMul_1MatMulzeros:output:01simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ

simple_rnn_cell_5/addAddV2"simple_rnn_cell_5/BiasAdd:output:0$simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ
k
simple_rnn_cell_5/ReluRelusimple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   ^
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
value	B : и
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_5_matmul_readvariableop_resource1simple_rnn_cell_5_biasadd_readvariableop_resource2simple_rnn_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_239595*
condR
while_cond_239594*8
output_shapes'
%: : : : :џџџџџџџџџ
: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ
*
element_dtype0*
num_elementsh
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
:џџџџџџџџџ
*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
Я
NoOpNoOp)^simple_rnn_cell_5/BiasAdd/ReadVariableOp(^simple_rnn_cell_5/MatMul/ReadVariableOp*^simple_rnn_cell_5/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2T
(simple_rnn_cell_5/BiasAdd/ReadVariableOp(simple_rnn_cell_5/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_5/MatMul/ReadVariableOp'simple_rnn_cell_5/MatMul/ReadVariableOp2V
)simple_rnn_cell_5/MatMul_1/ReadVariableOp)simple_rnn_cell_5/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
л-
Ш
while_body_241369
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_5_matmul_readvariableop_resource_0:
G
9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0:
L
:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0:


while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_5_matmul_readvariableop_resource:
E
7while_simple_rnn_cell_5_biasadd_readvariableop_resource:
J
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource:

Ђ.while/simple_rnn_cell_5/BiasAdd/ReadVariableOpЂ-while/simple_rnn_cell_5/MatMul/ReadVariableOpЂ/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0І
-while/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_5_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype0У
while/simple_rnn_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Є
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0*
_output_shapes
:
*
dtype0О
while/simple_rnn_cell_5/BiasAddBiasAdd(while/simple_rnn_cell_5/MatMul:product:06while/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Њ
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:

*
dtype0Њ
 while/simple_rnn_cell_5/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Ќ
while/simple_rnn_cell_5/addAddV2(while/simple_rnn_cell_5/BiasAdd:output:0*while/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ
w
while/simple_rnn_cell_5/ReluReluwhile/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ћ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0*while/simple_rnn_cell_5/Relu:activations:0*
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
: 
while/Identity_4Identity*while/simple_rnn_cell_5/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
п

while/NoOpNoOp/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_5_biasadd_readvariableop_resource9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_5_matmul_readvariableop_resource8while_simple_rnn_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ
: : : : : 2`
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_5/MatMul/ReadVariableOp-while/simple_rnn_cell_5/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
: 
ђ
А
3sequential_5_model_2_simple_rnn_6_while_cond_238457`
\sequential_5_model_2_simple_rnn_6_while_sequential_5_model_2_simple_rnn_6_while_loop_counterf
bsequential_5_model_2_simple_rnn_6_while_sequential_5_model_2_simple_rnn_6_while_maximum_iterations7
3sequential_5_model_2_simple_rnn_6_while_placeholder9
5sequential_5_model_2_simple_rnn_6_while_placeholder_19
5sequential_5_model_2_simple_rnn_6_while_placeholder_2b
^sequential_5_model_2_simple_rnn_6_while_less_sequential_5_model_2_simple_rnn_6_strided_slice_1x
tsequential_5_model_2_simple_rnn_6_while_sequential_5_model_2_simple_rnn_6_while_cond_238457___redundant_placeholder0x
tsequential_5_model_2_simple_rnn_6_while_sequential_5_model_2_simple_rnn_6_while_cond_238457___redundant_placeholder1x
tsequential_5_model_2_simple_rnn_6_while_sequential_5_model_2_simple_rnn_6_while_cond_238457___redundant_placeholder2x
tsequential_5_model_2_simple_rnn_6_while_sequential_5_model_2_simple_rnn_6_while_cond_238457___redundant_placeholder34
0sequential_5_model_2_simple_rnn_6_while_identity
ъ
,sequential_5/model_2/simple_rnn_6/while/LessLess3sequential_5_model_2_simple_rnn_6_while_placeholder^sequential_5_model_2_simple_rnn_6_while_less_sequential_5_model_2_simple_rnn_6_strided_slice_1*
T0*
_output_shapes
: 
0sequential_5/model_2/simple_rnn_6/while/IdentityIdentity0sequential_5/model_2/simple_rnn_6/while/Less:z:0*
T0
*
_output_shapes
: "m
0sequential_5_model_2_simple_rnn_6_while_identity9sequential_5/model_2/simple_rnn_6/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :џџџџџџџџџ: ::::: 
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
з
L
0__inference_repeat_vector_2_layer_call_fn_241551

inputs
identityУ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_repeat_vector_2_layer_call_and_return_conditional_losses_238858m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџџџџџџџџџџ:X T
0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ъ
њ
C__inference_dense_2_layer_call_and_return_conditional_losses_239829

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Л
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : П
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ї
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџc
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
џ

(__inference_model_2_layer_call_fn_239409
input_6
unknown:

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:
	unknown_4:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_239394s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_6
к
Њ
while_cond_239463
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_239463___redundant_placeholder04
0while_while_cond_239463___redundant_placeholder14
0while_while_cond_239463___redundant_placeholder24
0while_while_cond_239463___redundant_placeholder3
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
-: : : : :џџџџџџџџџ: ::::: 
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
к
Њ
while_cond_241968
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_241968___redundant_placeholder04
0while_while_cond_241968___redundant_placeholder14
0while_while_cond_241968___redundant_placeholder24
0while_while_cond_241968___redundant_placeholder3
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
-: : : : :џџџџџџџџџ: ::::: 
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:

З
-__inference_simple_rnn_6_layer_call_fn_241592

inputs
unknown:

	unknown_0:
	unknown_1:
identityЂStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_simple_rnn_6_layer_call_and_return_conditional_losses_239385s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameinputs
я	
Ч
-__inference_sequential_5_layer_call_fn_239950
model_2_input
unknown:

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identityЂStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallmodel_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_239910s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:џџџџџџџџџ
'
_user_specified_namemodel_2_input
П,
Ш
while_body_241753
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_6_matmul_readvariableop_resource_0:
G
9while_simple_rnn_cell_6_biasadd_readvariableop_resource_0:L
:while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_6_matmul_readvariableop_resource:
E
7while_simple_rnn_cell_6_biasadd_readvariableop_resource:J
8while_simple_rnn_cell_6_matmul_1_readvariableop_resource:Ђ.while/simple_rnn_cell_6/BiasAdd/ReadVariableOpЂ-while/simple_rnn_cell_6/MatMul/ReadVariableOpЂ/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ
*
element_dtype0І
-while/simple_rnn_cell_6/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_6_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype0У
while/simple_rnn_cell_6/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЄ
.while/simple_rnn_cell_6/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_6_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0О
while/simple_rnn_cell_6/BiasAddBiasAdd(while/simple_rnn_cell_6/MatMul:product:06while/simple_rnn_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЊ
/while/simple_rnn_cell_6/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0Њ
 while/simple_rnn_cell_6/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЌ
while/simple_rnn_cell_6/addAddV2(while/simple_rnn_cell_6/BiasAdd:output:0*while/simple_rnn_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџw
while/simple_rnn_cell_6/ReluReluwhile/simple_rnn_cell_6/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџг
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder*while/simple_rnn_cell_6/Relu:activations:0*
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
: 
while/Identity_4Identity*while/simple_rnn_cell_6/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџп

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
%: : : : :џџџџџџџџџ: : : : : 2`
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
л-
Ш
while_body_239595
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_5_matmul_readvariableop_resource_0:
G
9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0:
L
:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0:


while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_5_matmul_readvariableop_resource:
E
7while_simple_rnn_cell_5_biasadd_readvariableop_resource:
J
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource:

Ђ.while/simple_rnn_cell_5/BiasAdd/ReadVariableOpЂ-while/simple_rnn_cell_5/MatMul/ReadVariableOpЂ/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0І
-while/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_5_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype0У
while/simple_rnn_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Є
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0*
_output_shapes
:
*
dtype0О
while/simple_rnn_cell_5/BiasAddBiasAdd(while/simple_rnn_cell_5/MatMul:product:06while/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Њ
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:

*
dtype0Њ
 while/simple_rnn_cell_5/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Ќ
while/simple_rnn_cell_5/addAddV2(while/simple_rnn_cell_5/BiasAdd:output:0*while/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ
w
while/simple_rnn_cell_5/ReluReluwhile/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ћ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0*while/simple_rnn_cell_5/Relu:activations:0*
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
: 
while/Identity_4Identity*while/simple_rnn_cell_5/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
п

while/NoOpNoOp/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_5_biasadd_readvariableop_resource9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_5_matmul_readvariableop_resource8while_simple_rnn_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ
: : : : : 2`
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_5/MatMul/ReadVariableOp-while/simple_rnn_cell_5/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
: 
к
Њ
while_cond_241752
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_241752___redundant_placeholder04
0while_while_cond_241752___redundant_placeholder14
0while_while_cond_241752___redundant_placeholder24
0while_while_cond_241752___redundant_placeholder3
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
-: : : : :џџџџџџџџџ: ::::: 
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
к
Њ
while_cond_238772
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_238772___redundant_placeholder04
0while_while_cond_238772___redundant_placeholder14
0while_while_cond_238772___redundant_placeholder24
0while_while_cond_238772___redundant_placeholder3
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
-: : : : :џџџџџџџџџ
: ::::: 
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
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
:
я	
Ч
-__inference_sequential_5_layer_call_fn_239855
model_2_input
unknown:

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identityЂStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallmodel_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_239836s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:џџџџџџџџџ
'
_user_specified_namemodel_2_input
к
Њ
while_cond_239318
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_239318___redundant_placeholder04
0while_while_cond_239318___redundant_placeholder14
0while_while_cond_239318___redundant_placeholder24
0while_while_cond_239318___redundant_placeholder3
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
-: : : : :џџџџџџџџџ: ::::: 
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
K

__inference__traced_save_242281
file_prefix-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableopD
@savev2_simple_rnn_5_simple_rnn_cell_5_kernel_read_readvariableopN
Jsavev2_simple_rnn_5_simple_rnn_cell_5_recurrent_kernel_read_readvariableopB
>savev2_simple_rnn_5_simple_rnn_cell_5_bias_read_readvariableopD
@savev2_simple_rnn_6_simple_rnn_cell_6_kernel_read_readvariableopN
Jsavev2_simple_rnn_6_simple_rnn_cell_6_recurrent_kernel_read_readvariableopB
>savev2_simple_rnn_6_simple_rnn_cell_6_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableopK
Gsavev2_adam_simple_rnn_5_simple_rnn_cell_5_kernel_m_read_readvariableopU
Qsavev2_adam_simple_rnn_5_simple_rnn_cell_5_recurrent_kernel_m_read_readvariableopI
Esavev2_adam_simple_rnn_5_simple_rnn_cell_5_bias_m_read_readvariableopK
Gsavev2_adam_simple_rnn_6_simple_rnn_cell_6_kernel_m_read_readvariableopU
Qsavev2_adam_simple_rnn_6_simple_rnn_cell_6_recurrent_kernel_m_read_readvariableopI
Esavev2_adam_simple_rnn_6_simple_rnn_cell_6_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableopK
Gsavev2_adam_simple_rnn_5_simple_rnn_cell_5_kernel_v_read_readvariableopU
Qsavev2_adam_simple_rnn_5_simple_rnn_cell_5_recurrent_kernel_v_read_readvariableopI
Esavev2_adam_simple_rnn_5_simple_rnn_cell_5_bias_v_read_readvariableopK
Gsavev2_adam_simple_rnn_6_simple_rnn_cell_6_kernel_v_read_readvariableopU
Qsavev2_adam_simple_rnn_6_simple_rnn_cell_6_recurrent_kernel_v_read_readvariableopI
Esavev2_adam_simple_rnn_6_simple_rnn_cell_6_bias_v_read_readvariableop
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
: Ё
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*Ъ
valueРBН"B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHБ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B щ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop@savev2_simple_rnn_5_simple_rnn_cell_5_kernel_read_readvariableopJsavev2_simple_rnn_5_simple_rnn_cell_5_recurrent_kernel_read_readvariableop>savev2_simple_rnn_5_simple_rnn_cell_5_bias_read_readvariableop@savev2_simple_rnn_6_simple_rnn_cell_6_kernel_read_readvariableopJsavev2_simple_rnn_6_simple_rnn_cell_6_recurrent_kernel_read_readvariableop>savev2_simple_rnn_6_simple_rnn_cell_6_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableopGsavev2_adam_simple_rnn_5_simple_rnn_cell_5_kernel_m_read_readvariableopQsavev2_adam_simple_rnn_5_simple_rnn_cell_5_recurrent_kernel_m_read_readvariableopEsavev2_adam_simple_rnn_5_simple_rnn_cell_5_bias_m_read_readvariableopGsavev2_adam_simple_rnn_6_simple_rnn_cell_6_kernel_m_read_readvariableopQsavev2_adam_simple_rnn_6_simple_rnn_cell_6_recurrent_kernel_m_read_readvariableopEsavev2_adam_simple_rnn_6_simple_rnn_cell_6_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableopGsavev2_adam_simple_rnn_5_simple_rnn_cell_5_kernel_v_read_readvariableopQsavev2_adam_simple_rnn_5_simple_rnn_cell_5_recurrent_kernel_v_read_readvariableopEsavev2_adam_simple_rnn_5_simple_rnn_cell_5_bias_v_read_readvariableopGsavev2_adam_simple_rnn_6_simple_rnn_cell_6_kernel_v_read_readvariableopQsavev2_adam_simple_rnn_6_simple_rnn_cell_6_recurrent_kernel_v_read_readvariableopEsavev2_adam_simple_rnn_6_simple_rnn_cell_6_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	
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

identity_1Identity_1:output:0*ї
_input_shapesх
т: :::
:

:
:
::: : : : : : : : : :::
:

:
:
:::::
:

:
:
::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:
:$ 

_output_shapes

:

: 

_output_shapes
:
:$ 

_output_shapes

:
:$ 

_output_shapes

:: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:
:$ 

_output_shapes

:

: 

_output_shapes
:
:$ 

_output_shapes

:
:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:
:$ 

_output_shapes

:

: 

_output_shapes
:
:$ 

_output_shapes

:
:$  

_output_shapes

:: !

_output_shapes
::"

_output_shapes
: 


"__inference__traced_restore_242390
file_prefix1
assignvariableop_dense_2_kernel:-
assignvariableop_1_dense_2_bias:J
8assignvariableop_2_simple_rnn_5_simple_rnn_cell_5_kernel:
T
Bassignvariableop_3_simple_rnn_5_simple_rnn_cell_5_recurrent_kernel:

D
6assignvariableop_4_simple_rnn_5_simple_rnn_cell_5_bias:
J
8assignvariableop_5_simple_rnn_6_simple_rnn_cell_6_kernel:
T
Bassignvariableop_6_simple_rnn_6_simple_rnn_cell_6_recurrent_kernel:D
6assignvariableop_7_simple_rnn_6_simple_rnn_cell_6_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: %
assignvariableop_13_total_1: %
assignvariableop_14_count_1: #
assignvariableop_15_total: #
assignvariableop_16_count: ;
)assignvariableop_17_adam_dense_2_kernel_m:5
'assignvariableop_18_adam_dense_2_bias_m:R
@assignvariableop_19_adam_simple_rnn_5_simple_rnn_cell_5_kernel_m:
\
Jassignvariableop_20_adam_simple_rnn_5_simple_rnn_cell_5_recurrent_kernel_m:

L
>assignvariableop_21_adam_simple_rnn_5_simple_rnn_cell_5_bias_m:
R
@assignvariableop_22_adam_simple_rnn_6_simple_rnn_cell_6_kernel_m:
\
Jassignvariableop_23_adam_simple_rnn_6_simple_rnn_cell_6_recurrent_kernel_m:L
>assignvariableop_24_adam_simple_rnn_6_simple_rnn_cell_6_bias_m:;
)assignvariableop_25_adam_dense_2_kernel_v:5
'assignvariableop_26_adam_dense_2_bias_v:R
@assignvariableop_27_adam_simple_rnn_5_simple_rnn_cell_5_kernel_v:
\
Jassignvariableop_28_adam_simple_rnn_5_simple_rnn_cell_5_recurrent_kernel_v:

L
>assignvariableop_29_adam_simple_rnn_5_simple_rnn_cell_5_bias_v:
R
@assignvariableop_30_adam_simple_rnn_6_simple_rnn_cell_6_kernel_v:
\
Jassignvariableop_31_adam_simple_rnn_6_simple_rnn_cell_6_recurrent_kernel_v:L
>assignvariableop_32_adam_simple_rnn_6_simple_rnn_cell_6_bias_v:
identity_34ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Є
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*Ъ
valueРBН"B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHД
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ы
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_dense_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_2_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_2AssignVariableOp8assignvariableop_2_simple_rnn_5_simple_rnn_cell_5_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_3AssignVariableOpBassignvariableop_3_simple_rnn_5_simple_rnn_cell_5_recurrent_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_4AssignVariableOp6assignvariableop_4_simple_rnn_5_simple_rnn_cell_5_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_5AssignVariableOp8assignvariableop_5_simple_rnn_6_simple_rnn_cell_6_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_6AssignVariableOpBassignvariableop_6_simple_rnn_6_simple_rnn_cell_6_recurrent_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_7AssignVariableOp6assignvariableop_7_simple_rnn_6_simple_rnn_cell_6_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_dense_2_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_dense_2_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_19AssignVariableOp@assignvariableop_19_adam_simple_rnn_5_simple_rnn_cell_5_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_20AssignVariableOpJassignvariableop_20_adam_simple_rnn_5_simple_rnn_cell_5_recurrent_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_21AssignVariableOp>assignvariableop_21_adam_simple_rnn_5_simple_rnn_cell_5_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_22AssignVariableOp@assignvariableop_22_adam_simple_rnn_6_simple_rnn_cell_6_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_23AssignVariableOpJassignvariableop_23_adam_simple_rnn_6_simple_rnn_cell_6_recurrent_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_24AssignVariableOp>assignvariableop_24_adam_simple_rnn_6_simple_rnn_cell_6_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_dense_2_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_dense_2_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_27AssignVariableOp@assignvariableop_27_adam_simple_rnn_5_simple_rnn_cell_5_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_28AssignVariableOpJassignvariableop_28_adam_simple_rnn_5_simple_rnn_cell_5_recurrent_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_29AssignVariableOp>assignvariableop_29_adam_simple_rnn_5_simple_rnn_cell_5_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_30AssignVariableOp@assignvariableop_30_adam_simple_rnn_6_simple_rnn_cell_6_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_31AssignVariableOpJassignvariableop_31_adam_simple_rnn_6_simple_rnn_cell_6_recurrent_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_32AssignVariableOp>assignvariableop_32_adam_simple_rnn_6_simple_rnn_cell_6_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ѕ
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_34IdentityIdentity_33:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_34Identity_34:output:0*W
_input_shapesF
D: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_32AssignVariableOp_322(
AssignVariableOp_4AssignVariableOp_42(
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
г8
Ю
simple_rnn_6_while_body_2409576
2simple_rnn_6_while_simple_rnn_6_while_loop_counter<
8simple_rnn_6_while_simple_rnn_6_while_maximum_iterations"
simple_rnn_6_while_placeholder$
 simple_rnn_6_while_placeholder_1$
 simple_rnn_6_while_placeholder_25
1simple_rnn_6_while_simple_rnn_6_strided_slice_1_0q
msimple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor_0W
Esimple_rnn_6_while_simple_rnn_cell_6_matmul_readvariableop_resource_0:
T
Fsimple_rnn_6_while_simple_rnn_cell_6_biasadd_readvariableop_resource_0:Y
Gsimple_rnn_6_while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0:
simple_rnn_6_while_identity!
simple_rnn_6_while_identity_1!
simple_rnn_6_while_identity_2!
simple_rnn_6_while_identity_3!
simple_rnn_6_while_identity_43
/simple_rnn_6_while_simple_rnn_6_strided_slice_1o
ksimple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_6_tensorarrayunstack_tensorlistfromtensorU
Csimple_rnn_6_while_simple_rnn_cell_6_matmul_readvariableop_resource:
R
Dsimple_rnn_6_while_simple_rnn_cell_6_biasadd_readvariableop_resource:W
Esimple_rnn_6_while_simple_rnn_cell_6_matmul_1_readvariableop_resource:Ђ;simple_rnn_6/while/simple_rnn_cell_6/BiasAdd/ReadVariableOpЂ:simple_rnn_6/while/simple_rnn_cell_6/MatMul/ReadVariableOpЂ<simple_rnn_6/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp
Dsimple_rnn_6/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   ч
6simple_rnn_6/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_6_while_placeholderMsimple_rnn_6/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ
*
element_dtype0Р
:simple_rnn_6/while/simple_rnn_cell_6/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_6_while_simple_rnn_cell_6_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype0ъ
+simple_rnn_6/while/simple_rnn_cell_6/MatMulMatMul=simple_rnn_6/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_6/while/simple_rnn_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџО
;simple_rnn_6/while/simple_rnn_cell_6/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_6_while_simple_rnn_cell_6_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0х
,simple_rnn_6/while/simple_rnn_cell_6/BiasAddBiasAdd5simple_rnn_6/while/simple_rnn_cell_6/MatMul:product:0Csimple_rnn_6/while/simple_rnn_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџФ
<simple_rnn_6/while/simple_rnn_cell_6/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_6_while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0б
-simple_rnn_6/while/simple_rnn_cell_6/MatMul_1MatMul simple_rnn_6_while_placeholder_2Dsimple_rnn_6/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџг
(simple_rnn_6/while/simple_rnn_cell_6/addAddV25simple_rnn_6/while/simple_rnn_cell_6/BiasAdd:output:07simple_rnn_6/while/simple_rnn_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ
)simple_rnn_6/while/simple_rnn_cell_6/ReluRelu,simple_rnn_6/while/simple_rnn_cell_6/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
7simple_rnn_6/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_6_while_placeholder_1simple_rnn_6_while_placeholder7simple_rnn_6/while/simple_rnn_cell_6/Relu:activations:0*
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
: Ў
simple_rnn_6/while/Identity_4Identity7simple_rnn_6/while/simple_rnn_cell_6/Relu:activations:0^simple_rnn_6/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
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
%: : : : :џџџџџџџџџ: : : : : 2z
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
Ђ=
Л
H__inference_simple_rnn_6_layer_call_and_return_conditional_losses_239385

inputsB
0simple_rnn_cell_6_matmul_readvariableop_resource:
?
1simple_rnn_cell_6_biasadd_readvariableop_resource:D
2simple_rnn_cell_6_matmul_1_readvariableop_resource:
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
value	B :s
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
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
D
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
valueB"џџџџ
   р
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ
*
shrink_axis_mask
'simple_rnn_cell_6/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_6_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
simple_rnn_cell_6/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
(simple_rnn_cell_6/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ќ
simple_rnn_cell_6/BiasAddBiasAdd"simple_rnn_cell_6/MatMul:product:00simple_rnn_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
)simple_rnn_cell_6/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_6_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0
simple_rnn_cell_6/MatMul_1MatMulzeros:output:01simple_rnn_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_6/addAddV2"simple_rnn_cell_6/BiasAdd:output:0$simple_rnn_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџk
simple_rnn_cell_6/ReluRelusimple_rnn_cell_6/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
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
value	B : и
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_6_matmul_readvariableop_resource1simple_rnn_cell_6_biasadd_readvariableop_resource2simple_rnn_cell_6_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_239319*
condR
while_cond_239318*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Т
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
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
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџb
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџЯ
NoOpNoOp)^simple_rnn_cell_6/BiasAdd/ReadVariableOp(^simple_rnn_cell_6/MatMul/ReadVariableOp*^simple_rnn_cell_6/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ
: : : 2T
(simple_rnn_cell_6/BiasAdd/ReadVariableOp(simple_rnn_cell_6/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_6/MatMul/ReadVariableOp'simple_rnn_cell_6/MatMul/ReadVariableOp2V
)simple_rnn_cell_6/MatMul_1/ReadVariableOp)simple_rnn_cell_6/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameinputs
"
б
while_body_238612
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
 while_simple_rnn_cell_5_238634_0:
.
 while_simple_rnn_cell_5_238636_0:
2
 while_simple_rnn_cell_5_238638_0:


while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
while_simple_rnn_cell_5_238634:
,
while_simple_rnn_cell_5_238636:
0
while_simple_rnn_cell_5_238638:

Ђ/while/simple_rnn_cell_5/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0Ђ
/while/simple_rnn_cell_5/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2 while_simple_rnn_cell_5_238634_0 while_simple_rnn_cell_5_238636_0 while_simple_rnn_cell_5_238638_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ
:џџџџџџџџџ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_238598r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:08while/simple_rnn_cell_5/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity8while/simple_rnn_cell_5/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
~

while/NoOpNoOp0^while/simple_rnn_cell_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"B
while_simple_rnn_cell_5_238634 while_simple_rnn_cell_5_238634_0"B
while_simple_rnn_cell_5_238636 while_simple_rnn_cell_5_238636_0"B
while_simple_rnn_cell_5_238638 while_simple_rnn_cell_5_238638_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ
: : : : : 2b
/while/simple_rnn_cell_5/StatefulPartitionedCall/while/simple_rnn_cell_5/StatefulPartitionedCall: 
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
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
: 
К

Ё
simple_rnn_5_while_cond_2408476
2simple_rnn_5_while_simple_rnn_5_while_loop_counter<
8simple_rnn_5_while_simple_rnn_5_while_maximum_iterations"
simple_rnn_5_while_placeholder$
 simple_rnn_5_while_placeholder_1$
 simple_rnn_5_while_placeholder_28
4simple_rnn_5_while_less_simple_rnn_5_strided_slice_1N
Jsimple_rnn_5_while_simple_rnn_5_while_cond_240847___redundant_placeholder0N
Jsimple_rnn_5_while_simple_rnn_5_while_cond_240847___redundant_placeholder1N
Jsimple_rnn_5_while_simple_rnn_5_while_cond_240847___redundant_placeholder2N
Jsimple_rnn_5_while_simple_rnn_5_while_cond_240847___redundant_placeholder3
simple_rnn_5_while_identity

simple_rnn_5/while/LessLesssimple_rnn_5_while_placeholder4simple_rnn_5_while_less_simple_rnn_5_strided_slice_1*
T0*
_output_shapes
: e
simple_rnn_5/while/IdentityIdentitysimple_rnn_5/while/Less:z:0*
T0
*
_output_shapes
: "C
simple_rnn_5_while_identity$simple_rnn_5/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :џџџџџџџџџ
: ::::: 
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
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
:
ќ

(__inference_model_2_layer_call_fn_240587

inputs
unknown:

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:
	unknown_4:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_239709s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
л-
Ш
while_body_239202
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_5_matmul_readvariableop_resource_0:
G
9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0:
L
:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0:


while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_5_matmul_readvariableop_resource:
E
7while_simple_rnn_cell_5_biasadd_readvariableop_resource:
J
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource:

Ђ.while/simple_rnn_cell_5/BiasAdd/ReadVariableOpЂ-while/simple_rnn_cell_5/MatMul/ReadVariableOpЂ/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0І
-while/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_5_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype0У
while/simple_rnn_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Є
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0*
_output_shapes
:
*
dtype0О
while/simple_rnn_cell_5/BiasAddBiasAdd(while/simple_rnn_cell_5/MatMul:product:06while/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Њ
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:

*
dtype0Њ
 while/simple_rnn_cell_5/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Ќ
while/simple_rnn_cell_5/addAddV2(while/simple_rnn_cell_5/BiasAdd:output:0*while/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ
w
while/simple_rnn_cell_5/ReluReluwhile/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ћ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0*while/simple_rnn_cell_5/Relu:activations:0*
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
: 
while/Identity_4Identity*while/simple_rnn_cell_5/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
п

while/NoOpNoOp/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_5_biasadd_readvariableop_resource9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_5_matmul_readvariableop_resource8while_simple_rnn_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ
: : : : : 2`
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_5/MatMul/ReadVariableOp-while/simple_rnn_cell_5/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
: 
"
б
while_body_238773
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
 while_simple_rnn_cell_5_238795_0:
.
 while_simple_rnn_cell_5_238797_0:
2
 while_simple_rnn_cell_5_238799_0:


while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
while_simple_rnn_cell_5_238795:
,
while_simple_rnn_cell_5_238797:
0
while_simple_rnn_cell_5_238799:

Ђ/while/simple_rnn_cell_5/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0Ђ
/while/simple_rnn_cell_5/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2 while_simple_rnn_cell_5_238795_0 while_simple_rnn_cell_5_238797_0 while_simple_rnn_cell_5_238799_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ
:џџџџџџџџџ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_238720r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:08while/simple_rnn_cell_5/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity8while/simple_rnn_cell_5/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
~

while/NoOpNoOp0^while/simple_rnn_cell_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"B
while_simple_rnn_cell_5_238795 while_simple_rnn_cell_5_238795_0"B
while_simple_rnn_cell_5_238797 while_simple_rnn_cell_5_238797_0"B
while_simple_rnn_cell_5_238799 while_simple_rnn_cell_5_238799_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ
: : : : : 2b
/while/simple_rnn_cell_5/StatefulPartitionedCall/while/simple_rnn_cell_5/StatefulPartitionedCall: 
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
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
: 
їт
З
!__inference__wrapped_model_238550
model_2_inputd
Rsequential_5_model_2_simple_rnn_5_simple_rnn_cell_5_matmul_readvariableop_resource:
a
Ssequential_5_model_2_simple_rnn_5_simple_rnn_cell_5_biasadd_readvariableop_resource:
f
Tsequential_5_model_2_simple_rnn_5_simple_rnn_cell_5_matmul_1_readvariableop_resource:

d
Rsequential_5_model_2_simple_rnn_6_simple_rnn_cell_6_matmul_readvariableop_resource:
a
Ssequential_5_model_2_simple_rnn_6_simple_rnn_cell_6_biasadd_readvariableop_resource:f
Tsequential_5_model_2_simple_rnn_6_simple_rnn_cell_6_matmul_1_readvariableop_resource:H
6sequential_5_dense_2_tensordot_readvariableop_resource:B
4sequential_5_dense_2_biasadd_readvariableop_resource:
identityЂ+sequential_5/dense_2/BiasAdd/ReadVariableOpЂ-sequential_5/dense_2/Tensordot/ReadVariableOpЂJsequential_5/model_2/simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOpЂIsequential_5/model_2/simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOpЂKsequential_5/model_2/simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOpЂ'sequential_5/model_2/simple_rnn_5/whileЂJsequential_5/model_2/simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOpЂIsequential_5/model_2/simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOpЂKsequential_5/model_2/simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOpЂ'sequential_5/model_2/simple_rnn_6/whiled
'sequential_5/model_2/simple_rnn_5/ShapeShapemodel_2_input*
T0*
_output_shapes
:
5sequential_5/model_2/simple_rnn_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
7sequential_5/model_2/simple_rnn_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
7sequential_5/model_2/simple_rnn_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ћ
/sequential_5/model_2/simple_rnn_5/strided_sliceStridedSlice0sequential_5/model_2/simple_rnn_5/Shape:output:0>sequential_5/model_2/simple_rnn_5/strided_slice/stack:output:0@sequential_5/model_2/simple_rnn_5/strided_slice/stack_1:output:0@sequential_5/model_2/simple_rnn_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
0sequential_5/model_2/simple_rnn_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
й
.sequential_5/model_2/simple_rnn_5/zeros/packedPack8sequential_5/model_2/simple_rnn_5/strided_slice:output:09sequential_5/model_2/simple_rnn_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:r
-sequential_5/model_2/simple_rnn_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    в
'sequential_5/model_2/simple_rnn_5/zerosFill7sequential_5/model_2/simple_rnn_5/zeros/packed:output:06sequential_5/model_2/simple_rnn_5/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ

0sequential_5/model_2/simple_rnn_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          И
+sequential_5/model_2/simple_rnn_5/transpose	Transposemodel_2_input9sequential_5/model_2/simple_rnn_5/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
)sequential_5/model_2/simple_rnn_5/Shape_1Shape/sequential_5/model_2/simple_rnn_5/transpose:y:0*
T0*
_output_shapes
:
7sequential_5/model_2/simple_rnn_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9sequential_5/model_2/simple_rnn_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9sequential_5/model_2/simple_rnn_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1sequential_5/model_2/simple_rnn_5/strided_slice_1StridedSlice2sequential_5/model_2/simple_rnn_5/Shape_1:output:0@sequential_5/model_2/simple_rnn_5/strided_slice_1/stack:output:0Bsequential_5/model_2/simple_rnn_5/strided_slice_1/stack_1:output:0Bsequential_5/model_2/simple_rnn_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
=sequential_5/model_2/simple_rnn_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
/sequential_5/model_2/simple_rnn_5/TensorArrayV2TensorListReserveFsequential_5/model_2/simple_rnn_5/TensorArrayV2/element_shape:output:0:sequential_5/model_2/simple_rnn_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвЈ
Wsequential_5/model_2/simple_rnn_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ц
Isequential_5/model_2/simple_rnn_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor/sequential_5/model_2/simple_rnn_5/transpose:y:0`sequential_5/model_2/simple_rnn_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
7sequential_5/model_2/simple_rnn_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9sequential_5/model_2/simple_rnn_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9sequential_5/model_2/simple_rnn_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1sequential_5/model_2/simple_rnn_5/strided_slice_2StridedSlice/sequential_5/model_2/simple_rnn_5/transpose:y:0@sequential_5/model_2/simple_rnn_5/strided_slice_2/stack:output:0Bsequential_5/model_2/simple_rnn_5/strided_slice_2/stack_1:output:0Bsequential_5/model_2/simple_rnn_5/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskм
Isequential_5/model_2/simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOpRsequential_5_model_2_simple_rnn_5_simple_rnn_cell_5_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
:sequential_5/model_2/simple_rnn_5/simple_rnn_cell_5/MatMulMatMul:sequential_5/model_2/simple_rnn_5/strided_slice_2:output:0Qsequential_5/model_2/simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
к
Jsequential_5/model_2/simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOpSsequential_5_model_2_simple_rnn_5_simple_rnn_cell_5_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
;sequential_5/model_2/simple_rnn_5/simple_rnn_cell_5/BiasAddBiasAddDsequential_5/model_2/simple_rnn_5/simple_rnn_cell_5/MatMul:product:0Rsequential_5/model_2/simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
р
Ksequential_5/model_2/simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOpTsequential_5_model_2_simple_rnn_5_simple_rnn_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype0џ
<sequential_5/model_2/simple_rnn_5/simple_rnn_cell_5/MatMul_1MatMul0sequential_5/model_2/simple_rnn_5/zeros:output:0Ssequential_5/model_2/simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ

7sequential_5/model_2/simple_rnn_5/simple_rnn_cell_5/addAddV2Dsequential_5/model_2/simple_rnn_5/simple_rnn_cell_5/BiasAdd:output:0Fsequential_5/model_2/simple_rnn_5/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ
Џ
8sequential_5/model_2/simple_rnn_5/simple_rnn_cell_5/ReluRelu;sequential_5/model_2/simple_rnn_5/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ

?sequential_5/model_2/simple_rnn_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   
>sequential_5/model_2/simple_rnn_5/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Ћ
1sequential_5/model_2/simple_rnn_5/TensorArrayV2_1TensorListReserveHsequential_5/model_2/simple_rnn_5/TensorArrayV2_1/element_shape:output:0Gsequential_5/model_2/simple_rnn_5/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвh
&sequential_5/model_2/simple_rnn_5/timeConst*
_output_shapes
: *
dtype0*
value	B : 
:sequential_5/model_2/simple_rnn_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџv
4sequential_5/model_2/simple_rnn_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 	
'sequential_5/model_2/simple_rnn_5/whileWhile=sequential_5/model_2/simple_rnn_5/while/loop_counter:output:0Csequential_5/model_2/simple_rnn_5/while/maximum_iterations:output:0/sequential_5/model_2/simple_rnn_5/time:output:0:sequential_5/model_2/simple_rnn_5/TensorArrayV2_1:handle:00sequential_5/model_2/simple_rnn_5/zeros:output:0:sequential_5/model_2/simple_rnn_5/strided_slice_1:output:0Ysequential_5/model_2/simple_rnn_5/TensorArrayUnstack/TensorListFromTensor:output_handle:0Rsequential_5_model_2_simple_rnn_5_simple_rnn_cell_5_matmul_readvariableop_resourceSsequential_5_model_2_simple_rnn_5_simple_rnn_cell_5_biasadd_readvariableop_resourceTsequential_5_model_2_simple_rnn_5_simple_rnn_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *?
body7R5
3sequential_5_model_2_simple_rnn_5_while_body_238349*?
cond7R5
3sequential_5_model_2_simple_rnn_5_while_cond_238348*8
output_shapes'
%: : : : :џџџџџџџџџ
: : : : : *
parallel_iterations Ѓ
Rsequential_5/model_2/simple_rnn_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   М
Dsequential_5/model_2/simple_rnn_5/TensorArrayV2Stack/TensorListStackTensorListStack0sequential_5/model_2/simple_rnn_5/while:output:3[sequential_5/model_2/simple_rnn_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ
*
element_dtype0*
num_elements
7sequential_5/model_2/simple_rnn_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ
9sequential_5/model_2/simple_rnn_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
9sequential_5/model_2/simple_rnn_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Б
1sequential_5/model_2/simple_rnn_5/strided_slice_3StridedSliceMsequential_5/model_2/simple_rnn_5/TensorArrayV2Stack/TensorListStack:tensor:0@sequential_5/model_2/simple_rnn_5/strided_slice_3/stack:output:0Bsequential_5/model_2/simple_rnn_5/strided_slice_3/stack_1:output:0Bsequential_5/model_2/simple_rnn_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ
*
shrink_axis_mask
2sequential_5/model_2/simple_rnn_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
-sequential_5/model_2/simple_rnn_5/transpose_1	TransposeMsequential_5/model_2/simple_rnn_5/TensorArrayV2Stack/TensorListStack:tensor:0;sequential_5/model_2/simple_rnn_5/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
u
3sequential_5/model_2/repeat_vector_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :э
/sequential_5/model_2/repeat_vector_2/ExpandDims
ExpandDims:sequential_5/model_2/simple_rnn_5/strided_slice_3:output:0<sequential_5/model_2/repeat_vector_2/ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ

*sequential_5/model_2/repeat_vector_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"         ж
)sequential_5/model_2/repeat_vector_2/TileTile8sequential_5/model_2/repeat_vector_2/ExpandDims:output:03sequential_5/model_2/repeat_vector_2/stack:output:0*
T0*+
_output_shapes
:џџџџџџџџџ

'sequential_5/model_2/simple_rnn_6/ShapeShape2sequential_5/model_2/repeat_vector_2/Tile:output:0*
T0*
_output_shapes
:
5sequential_5/model_2/simple_rnn_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
7sequential_5/model_2/simple_rnn_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
7sequential_5/model_2/simple_rnn_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ћ
/sequential_5/model_2/simple_rnn_6/strided_sliceStridedSlice0sequential_5/model_2/simple_rnn_6/Shape:output:0>sequential_5/model_2/simple_rnn_6/strided_slice/stack:output:0@sequential_5/model_2/simple_rnn_6/strided_slice/stack_1:output:0@sequential_5/model_2/simple_rnn_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
0sequential_5/model_2/simple_rnn_6/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :й
.sequential_5/model_2/simple_rnn_6/zeros/packedPack8sequential_5/model_2/simple_rnn_6/strided_slice:output:09sequential_5/model_2/simple_rnn_6/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:r
-sequential_5/model_2/simple_rnn_6/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    в
'sequential_5/model_2/simple_rnn_6/zerosFill7sequential_5/model_2/simple_rnn_6/zeros/packed:output:06sequential_5/model_2/simple_rnn_6/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
0sequential_5/model_2/simple_rnn_6/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          н
+sequential_5/model_2/simple_rnn_6/transpose	Transpose2sequential_5/model_2/repeat_vector_2/Tile:output:09sequential_5/model_2/simple_rnn_6/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ

)sequential_5/model_2/simple_rnn_6/Shape_1Shape/sequential_5/model_2/simple_rnn_6/transpose:y:0*
T0*
_output_shapes
:
7sequential_5/model_2/simple_rnn_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9sequential_5/model_2/simple_rnn_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9sequential_5/model_2/simple_rnn_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1sequential_5/model_2/simple_rnn_6/strided_slice_1StridedSlice2sequential_5/model_2/simple_rnn_6/Shape_1:output:0@sequential_5/model_2/simple_rnn_6/strided_slice_1/stack:output:0Bsequential_5/model_2/simple_rnn_6/strided_slice_1/stack_1:output:0Bsequential_5/model_2/simple_rnn_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
=sequential_5/model_2/simple_rnn_6/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
/sequential_5/model_2/simple_rnn_6/TensorArrayV2TensorListReserveFsequential_5/model_2/simple_rnn_6/TensorArrayV2/element_shape:output:0:sequential_5/model_2/simple_rnn_6/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвЈ
Wsequential_5/model_2/simple_rnn_6/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   Ц
Isequential_5/model_2/simple_rnn_6/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor/sequential_5/model_2/simple_rnn_6/transpose:y:0`sequential_5/model_2/simple_rnn_6/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
7sequential_5/model_2/simple_rnn_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9sequential_5/model_2/simple_rnn_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9sequential_5/model_2/simple_rnn_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1sequential_5/model_2/simple_rnn_6/strided_slice_2StridedSlice/sequential_5/model_2/simple_rnn_6/transpose:y:0@sequential_5/model_2/simple_rnn_6/strided_slice_2/stack:output:0Bsequential_5/model_2/simple_rnn_6/strided_slice_2/stack_1:output:0Bsequential_5/model_2/simple_rnn_6/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ
*
shrink_axis_maskм
Isequential_5/model_2/simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOpReadVariableOpRsequential_5_model_2_simple_rnn_6_simple_rnn_cell_6_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
:sequential_5/model_2/simple_rnn_6/simple_rnn_cell_6/MatMulMatMul:sequential_5/model_2/simple_rnn_6/strided_slice_2:output:0Qsequential_5/model_2/simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџк
Jsequential_5/model_2/simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOpReadVariableOpSsequential_5_model_2_simple_rnn_6_simple_rnn_cell_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
;sequential_5/model_2/simple_rnn_6/simple_rnn_cell_6/BiasAddBiasAddDsequential_5/model_2/simple_rnn_6/simple_rnn_cell_6/MatMul:product:0Rsequential_5/model_2/simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџр
Ksequential_5/model_2/simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOpReadVariableOpTsequential_5_model_2_simple_rnn_6_simple_rnn_cell_6_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0џ
<sequential_5/model_2/simple_rnn_6/simple_rnn_cell_6/MatMul_1MatMul0sequential_5/model_2/simple_rnn_6/zeros:output:0Ssequential_5/model_2/simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
7sequential_5/model_2/simple_rnn_6/simple_rnn_cell_6/addAddV2Dsequential_5/model_2/simple_rnn_6/simple_rnn_cell_6/BiasAdd:output:0Fsequential_5/model_2/simple_rnn_6/simple_rnn_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџЏ
8sequential_5/model_2/simple_rnn_6/simple_rnn_cell_6/ReluRelu;sequential_5/model_2/simple_rnn_6/simple_rnn_cell_6/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
?sequential_5/model_2/simple_rnn_6/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
1sequential_5/model_2/simple_rnn_6/TensorArrayV2_1TensorListReserveHsequential_5/model_2/simple_rnn_6/TensorArrayV2_1/element_shape:output:0:sequential_5/model_2/simple_rnn_6/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвh
&sequential_5/model_2/simple_rnn_6/timeConst*
_output_shapes
: *
dtype0*
value	B : 
:sequential_5/model_2/simple_rnn_6/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџv
4sequential_5/model_2/simple_rnn_6/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 	
'sequential_5/model_2/simple_rnn_6/whileWhile=sequential_5/model_2/simple_rnn_6/while/loop_counter:output:0Csequential_5/model_2/simple_rnn_6/while/maximum_iterations:output:0/sequential_5/model_2/simple_rnn_6/time:output:0:sequential_5/model_2/simple_rnn_6/TensorArrayV2_1:handle:00sequential_5/model_2/simple_rnn_6/zeros:output:0:sequential_5/model_2/simple_rnn_6/strided_slice_1:output:0Ysequential_5/model_2/simple_rnn_6/TensorArrayUnstack/TensorListFromTensor:output_handle:0Rsequential_5_model_2_simple_rnn_6_simple_rnn_cell_6_matmul_readvariableop_resourceSsequential_5_model_2_simple_rnn_6_simple_rnn_cell_6_biasadd_readvariableop_resourceTsequential_5_model_2_simple_rnn_6_simple_rnn_cell_6_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *?
body7R5
3sequential_5_model_2_simple_rnn_6_while_body_238458*?
cond7R5
3sequential_5_model_2_simple_rnn_6_while_cond_238457*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations Ѓ
Rsequential_5/model_2/simple_rnn_6/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ј
Dsequential_5/model_2/simple_rnn_6/TensorArrayV2Stack/TensorListStackTensorListStack0sequential_5/model_2/simple_rnn_6/while:output:3[sequential_5/model_2/simple_rnn_6/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype0
7sequential_5/model_2/simple_rnn_6/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ
9sequential_5/model_2/simple_rnn_6/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
9sequential_5/model_2/simple_rnn_6/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Б
1sequential_5/model_2/simple_rnn_6/strided_slice_3StridedSliceMsequential_5/model_2/simple_rnn_6/TensorArrayV2Stack/TensorListStack:tensor:0@sequential_5/model_2/simple_rnn_6/strided_slice_3/stack:output:0Bsequential_5/model_2/simple_rnn_6/strided_slice_3/stack_1:output:0Bsequential_5/model_2/simple_rnn_6/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
2sequential_5/model_2/simple_rnn_6/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
-sequential_5/model_2/simple_rnn_6/transpose_1	TransposeMsequential_5/model_2/simple_rnn_6/TensorArrayV2Stack/TensorListStack:tensor:0;sequential_5/model_2/simple_rnn_6/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџЄ
-sequential_5/dense_2/Tensordot/ReadVariableOpReadVariableOp6sequential_5_dense_2_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0m
#sequential_5/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:t
#sequential_5/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
$sequential_5/dense_2/Tensordot/ShapeShape1sequential_5/model_2/simple_rnn_6/transpose_1:y:0*
T0*
_output_shapes
:n
,sequential_5/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
'sequential_5/dense_2/Tensordot/GatherV2GatherV2-sequential_5/dense_2/Tensordot/Shape:output:0,sequential_5/dense_2/Tensordot/free:output:05sequential_5/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
.sequential_5/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
)sequential_5/dense_2/Tensordot/GatherV2_1GatherV2-sequential_5/dense_2/Tensordot/Shape:output:0,sequential_5/dense_2/Tensordot/axes:output:07sequential_5/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
$sequential_5/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ­
#sequential_5/dense_2/Tensordot/ProdProd0sequential_5/dense_2/Tensordot/GatherV2:output:0-sequential_5/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&sequential_5/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Г
%sequential_5/dense_2/Tensordot/Prod_1Prod2sequential_5/dense_2/Tensordot/GatherV2_1:output:0/sequential_5/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*sequential_5/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : №
%sequential_5/dense_2/Tensordot/concatConcatV2,sequential_5/dense_2/Tensordot/free:output:0,sequential_5/dense_2/Tensordot/axes:output:03sequential_5/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:И
$sequential_5/dense_2/Tensordot/stackPack,sequential_5/dense_2/Tensordot/Prod:output:0.sequential_5/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ю
(sequential_5/dense_2/Tensordot/transpose	Transpose1sequential_5/model_2/simple_rnn_6/transpose_1:y:0.sequential_5/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџЩ
&sequential_5/dense_2/Tensordot/ReshapeReshape,sequential_5/dense_2/Tensordot/transpose:y:0-sequential_5/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЩ
%sequential_5/dense_2/Tensordot/MatMulMatMul/sequential_5/dense_2/Tensordot/Reshape:output:05sequential_5/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџp
&sequential_5/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:n
,sequential_5/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ћ
'sequential_5/dense_2/Tensordot/concat_1ConcatV20sequential_5/dense_2/Tensordot/GatherV2:output:0/sequential_5/dense_2/Tensordot/Const_2:output:05sequential_5/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Т
sequential_5/dense_2/TensordotReshape/sequential_5/dense_2/Tensordot/MatMul:product:00sequential_5/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
+sequential_5/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_5_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Л
sequential_5/dense_2/BiasAddBiasAdd'sequential_5/dense_2/Tensordot:output:03sequential_5/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџx
IdentityIdentity%sequential_5/dense_2/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџЦ
NoOpNoOp,^sequential_5/dense_2/BiasAdd/ReadVariableOp.^sequential_5/dense_2/Tensordot/ReadVariableOpK^sequential_5/model_2/simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOpJ^sequential_5/model_2/simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOpL^sequential_5/model_2/simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp(^sequential_5/model_2/simple_rnn_5/whileK^sequential_5/model_2/simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOpJ^sequential_5/model_2/simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOpL^sequential_5/model_2/simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOp(^sequential_5/model_2/simple_rnn_6/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : : : 2Z
+sequential_5/dense_2/BiasAdd/ReadVariableOp+sequential_5/dense_2/BiasAdd/ReadVariableOp2^
-sequential_5/dense_2/Tensordot/ReadVariableOp-sequential_5/dense_2/Tensordot/ReadVariableOp2
Jsequential_5/model_2/simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOpJsequential_5/model_2/simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp2
Isequential_5/model_2/simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOpIsequential_5/model_2/simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp2
Ksequential_5/model_2/simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOpKsequential_5/model_2/simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp2R
'sequential_5/model_2/simple_rnn_5/while'sequential_5/model_2/simple_rnn_5/while2
Jsequential_5/model_2/simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOpJsequential_5/model_2/simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOp2
Isequential_5/model_2/simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOpIsequential_5/model_2/simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOp2
Ksequential_5/model_2/simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOpKsequential_5/model_2/simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOp2R
'sequential_5/model_2/simple_rnn_6/while'sequential_5/model_2/simple_rnn_6/while:Z V
+
_output_shapes
:џџџџџџџџџ
'
_user_specified_namemodel_2_input
р=
Н
H__inference_simple_rnn_6_layer_call_and_return_conditional_losses_241711
inputs_0B
0simple_rnn_cell_6_matmul_readvariableop_resource:
?
1simple_rnn_cell_6_biasadd_readvariableop_resource:D
2simple_rnn_cell_6_matmul_1_readvariableop_resource:
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
value	B :s
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
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
D
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
valueB"џџџџ
   р
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ
*
shrink_axis_mask
'simple_rnn_cell_6/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_6_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
simple_rnn_cell_6/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
(simple_rnn_cell_6/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ќ
simple_rnn_cell_6/BiasAddBiasAdd"simple_rnn_cell_6/MatMul:product:00simple_rnn_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
)simple_rnn_cell_6/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_6_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0
simple_rnn_cell_6/MatMul_1MatMulzeros:output:01simple_rnn_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_6/addAddV2"simple_rnn_cell_6/BiasAdd:output:0$simple_rnn_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџk
simple_rnn_cell_6/ReluRelusimple_rnn_cell_6/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
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
value	B : и
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_6_matmul_readvariableop_resource1simple_rnn_cell_6_biasadd_readvariableop_resource2simple_rnn_cell_6_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_241645*
condR
while_cond_241644*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
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
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџk
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџЯ
NoOpNoOp)^simple_rnn_cell_6/BiasAdd/ReadVariableOp(^simple_rnn_cell_6/MatMul/ReadVariableOp*^simple_rnn_cell_6/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ
: : : 2T
(simple_rnn_cell_6/BiasAdd/ReadVariableOp(simple_rnn_cell_6/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_6/MatMul/ReadVariableOp'simple_rnn_cell_6/MatMul/ReadVariableOp2V
)simple_rnn_cell_6/MatMul_1/ReadVariableOp)simple_rnn_cell_6/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ

"
_user_specified_name
inputs/0

ъ
M__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_242080

inputs
states_00
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
2
 matmul_1_readvariableop_resource:


identity

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ
G
ReluReluadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
c

Identity_1IdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ

NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ:џџџџџџџџџ
: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ

"
_user_specified_name
states/0

ъ
M__inference_simple_rnn_cell_6_layer_call_and_return_conditional_losses_242159

inputs
states_00
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:2
 matmul_1_readvariableop_resource:
identity

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
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
:џџџџџџџџџx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџG
ReluReluadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџc

Identity_1IdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ
:џџџџџџџџџ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0
Ъ
њ
C__inference_dense_2_layer_call_and_return_conditional_losses_241062

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Л
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : П
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ї
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџc
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
П,
Ш
while_body_241969
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_6_matmul_readvariableop_resource_0:
G
9while_simple_rnn_cell_6_biasadd_readvariableop_resource_0:L
:while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_6_matmul_readvariableop_resource:
E
7while_simple_rnn_cell_6_biasadd_readvariableop_resource:J
8while_simple_rnn_cell_6_matmul_1_readvariableop_resource:Ђ.while/simple_rnn_cell_6/BiasAdd/ReadVariableOpЂ-while/simple_rnn_cell_6/MatMul/ReadVariableOpЂ/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ
*
element_dtype0І
-while/simple_rnn_cell_6/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_6_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype0У
while/simple_rnn_cell_6/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЄ
.while/simple_rnn_cell_6/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_6_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0О
while/simple_rnn_cell_6/BiasAddBiasAdd(while/simple_rnn_cell_6/MatMul:product:06while/simple_rnn_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЊ
/while/simple_rnn_cell_6/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0Њ
 while/simple_rnn_cell_6/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЌ
while/simple_rnn_cell_6/addAddV2(while/simple_rnn_cell_6/BiasAdd:output:0*while/simple_rnn_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџw
while/simple_rnn_cell_6/ReluReluwhile/simple_rnn_cell_6/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџг
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder*while/simple_rnn_cell_6/Relu:activations:0*
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
: 
while/Identity_4Identity*while/simple_rnn_cell_6/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџп

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
%: : : : :џџџџџџџџџ: : : : : 2`
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 

Ж
H__inference_sequential_5_layer_call_and_return_conditional_losses_239994
model_2_input 
model_2_239975:

model_2_239977:
 
model_2_239979:

 
model_2_239981:

model_2_239983: 
model_2_239985: 
dense_2_239988:
dense_2_239990:
identityЂdense_2/StatefulPartitionedCallЂmodel_2/StatefulPartitionedCallП
model_2/StatefulPartitionedCallStatefulPartitionedCallmodel_2_inputmodel_2_239975model_2_239977model_2_239979model_2_239981model_2_239983model_2_239985*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_239709
dense_2/StatefulPartitionedCallStatefulPartitionedCall(model_2/StatefulPartitionedCall:output:0dense_2_239988dense_2_239990*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_239829{
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ
NoOpNoOp ^dense_2/StatefulPartitionedCall ^model_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
model_2/StatefulPartitionedCallmodel_2/StatefulPartitionedCall:Z V
+
_output_shapes
:џџџџџџџџџ
'
_user_specified_namemodel_2_input
л-
Ш
while_body_241479
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_5_matmul_readvariableop_resource_0:
G
9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0:
L
:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0:


while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_5_matmul_readvariableop_resource:
E
7while_simple_rnn_cell_5_biasadd_readvariableop_resource:
J
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource:

Ђ.while/simple_rnn_cell_5/BiasAdd/ReadVariableOpЂ-while/simple_rnn_cell_5/MatMul/ReadVariableOpЂ/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0І
-while/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_5_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype0У
while/simple_rnn_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Є
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0*
_output_shapes
:
*
dtype0О
while/simple_rnn_cell_5/BiasAddBiasAdd(while/simple_rnn_cell_5/MatMul:product:06while/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Њ
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:

*
dtype0Њ
 while/simple_rnn_cell_5/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Ќ
while/simple_rnn_cell_5/addAddV2(while/simple_rnn_cell_5/BiasAdd:output:0*while/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ
w
while/simple_rnn_cell_5/ReluReluwhile/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ћ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0*while/simple_rnn_cell_5/Relu:activations:0*
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
: 
while/Identity_4Identity*while/simple_rnn_cell_5/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
п

while/NoOpNoOp/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_5_biasadd_readvariableop_resource9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_5_matmul_readvariableop_resource8while_simple_rnn_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ
: : : : : 2`
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_5/MatMul/ReadVariableOp-while/simple_rnn_cell_5/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
: 
к	
Р
-__inference_sequential_5_layer_call_fn_240065

inputs
unknown:

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identityЂStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_239910s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ђ=
Л
H__inference_simple_rnn_6_layer_call_and_return_conditional_losses_239530

inputsB
0simple_rnn_cell_6_matmul_readvariableop_resource:
?
1simple_rnn_cell_6_biasadd_readvariableop_resource:D
2simple_rnn_cell_6_matmul_1_readvariableop_resource:
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
value	B :s
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
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
D
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
valueB"џџџџ
   р
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ
*
shrink_axis_mask
'simple_rnn_cell_6/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_6_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
simple_rnn_cell_6/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
(simple_rnn_cell_6/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ќ
simple_rnn_cell_6/BiasAddBiasAdd"simple_rnn_cell_6/MatMul:product:00simple_rnn_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
)simple_rnn_cell_6/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_6_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0
simple_rnn_cell_6/MatMul_1MatMulzeros:output:01simple_rnn_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
simple_rnn_cell_6/addAddV2"simple_rnn_cell_6/BiasAdd:output:0$simple_rnn_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџk
simple_rnn_cell_6/ReluRelusimple_rnn_cell_6/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
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
value	B : и
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_6_matmul_readvariableop_resource1simple_rnn_cell_6_biasadd_readvariableop_resource2simple_rnn_cell_6_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_239464*
condR
while_cond_239463*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Т
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
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
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџb
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџЯ
NoOpNoOp)^simple_rnn_cell_6/BiasAdd/ReadVariableOp(^simple_rnn_cell_6/MatMul/ReadVariableOp*^simple_rnn_cell_6/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ
: : : 2T
(simple_rnn_cell_6/BiasAdd/ReadVariableOp(simple_rnn_cell_6/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_6/MatMul/ReadVariableOp'simple_rnn_cell_6/MatMul/ReadVariableOp2V
)simple_rnn_cell_6/MatMul_1/ReadVariableOp)simple_rnn_cell_6/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameinputs
Ы>
Н
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_241326
inputs_0B
0simple_rnn_cell_5_matmul_readvariableop_resource:
?
1simple_rnn_cell_5_biasadd_readvariableop_resource:
D
2simple_rnn_cell_5_matmul_1_readvariableop_resource:


identityЂ(simple_rnn_cell_5/BiasAdd/ReadVariableOpЂ'simple_rnn_cell_5/MatMul/ReadVariableOpЂ)simple_rnn_cell_5/MatMul_1/ReadVariableOpЂwhile=
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
value	B :
s
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
:џџџџџџџџџ
c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџD
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
valueB"џџџџ   р
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
'simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_5_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
simple_rnn_cell_5/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ

(simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_5_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Ќ
simple_rnn_cell_5/BiasAddBiasAdd"simple_rnn_cell_5/MatMul:product:00simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ

)simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype0
simple_rnn_cell_5/MatMul_1MatMulzeros:output:01simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ

simple_rnn_cell_5/addAddV2"simple_rnn_cell_5/BiasAdd:output:0$simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ
k
simple_rnn_cell_5/ReluRelusimple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   ^
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
value	B : и
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_5_matmul_readvariableop_resource1simple_rnn_cell_5_biasadd_readvariableop_resource2simple_rnn_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_241259*
condR
while_cond_241258*8
output_shapes'
%: : : : :џџџџџџџџџ
: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ
*
element_dtype0*
num_elementsh
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
:џџџџџџџџџ
*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
Я
NoOpNoOp)^simple_rnn_cell_5/BiasAdd/ReadVariableOp(^simple_rnn_cell_5/MatMul/ReadVariableOp*^simple_rnn_cell_5/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2T
(simple_rnn_cell_5/BiasAdd/ReadVariableOp(simple_rnn_cell_5/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_5/MatMul/ReadVariableOp'simple_rnn_cell_5/MatMul/ReadVariableOp2V
)simple_rnn_cell_5/MatMul_1/ReadVariableOp)simple_rnn_cell_5/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
л-
Ш
while_body_241259
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_5_matmul_readvariableop_resource_0:
G
9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0:
L
:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0:


while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_5_matmul_readvariableop_resource:
E
7while_simple_rnn_cell_5_biasadd_readvariableop_resource:
J
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource:

Ђ.while/simple_rnn_cell_5/BiasAdd/ReadVariableOpЂ-while/simple_rnn_cell_5/MatMul/ReadVariableOpЂ/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0І
-while/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_5_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype0У
while/simple_rnn_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Є
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0*
_output_shapes
:
*
dtype0О
while/simple_rnn_cell_5/BiasAddBiasAdd(while/simple_rnn_cell_5/MatMul:product:06while/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Њ
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:

*
dtype0Њ
 while/simple_rnn_cell_5/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Ќ
while/simple_rnn_cell_5/addAddV2(while/simple_rnn_cell_5/BiasAdd:output:0*while/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ
w
while/simple_rnn_cell_5/ReluReluwhile/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ћ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0*while/simple_rnn_cell_5/Relu:activations:0*
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
: 
while/Identity_4Identity*while/simple_rnn_cell_5/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
п

while/NoOpNoOp/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_5_biasadd_readvariableop_resource9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_5_matmul_readvariableop_resource8while_simple_rnn_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ
: : : : : 2`
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_5/MatMul/ReadVariableOp-while/simple_rnn_cell_5/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
: 
Ј>
Л
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_241436

inputsB
0simple_rnn_cell_5_matmul_readvariableop_resource:
?
1simple_rnn_cell_5_biasadd_readvariableop_resource:
D
2simple_rnn_cell_5_matmul_1_readvariableop_resource:


identityЂ(simple_rnn_cell_5/BiasAdd/ReadVariableOpЂ'simple_rnn_cell_5/MatMul/ReadVariableOpЂ)simple_rnn_cell_5/MatMul_1/ReadVariableOpЂwhile;
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
value	B :
s
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
:џџџџџџџџџ
c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџD
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
valueB"џџџџ   р
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
'simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_5_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
simple_rnn_cell_5/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ

(simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_5_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Ќ
simple_rnn_cell_5/BiasAddBiasAdd"simple_rnn_cell_5/MatMul:product:00simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ

)simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype0
simple_rnn_cell_5/MatMul_1MatMulzeros:output:01simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ

simple_rnn_cell_5/addAddV2"simple_rnn_cell_5/BiasAdd:output:0$simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ
k
simple_rnn_cell_5/ReluRelusimple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   ^
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
value	B : и
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_5_matmul_readvariableop_resource1simple_rnn_cell_5_biasadd_readvariableop_resource2simple_rnn_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_241369*
condR
while_cond_241368*8
output_shapes'
%: : : : :џџџџџџџџџ
: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ
*
element_dtype0*
num_elementsh
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
:џџџџџџџџџ
*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
Я
NoOpNoOp)^simple_rnn_cell_5/BiasAdd/ReadVariableOp(^simple_rnn_cell_5/MatMul/ReadVariableOp*^simple_rnn_cell_5/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2T
(simple_rnn_cell_5/BiasAdd/ReadVariableOp(simple_rnn_cell_5/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_5/MatMul/ReadVariableOp'simple_rnn_cell_5/MatMul/ReadVariableOp2V
)simple_rnn_cell_5/MatMul_1/ReadVariableOp)simple_rnn_cell_5/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

Й
-__inference_simple_rnn_5_layer_call_fn_241084
inputs_0
unknown:

	unknown_0:

	unknown_1:


identityЂStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_238837o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0

ш
M__inference_simple_rnn_cell_6_layer_call_and_return_conditional_losses_238909

inputs

states0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:2
 matmul_1_readvariableop_resource:
identity

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
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
:џџџџџџџџџx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџG
ReluReluadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџc

Identity_1IdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ
:џџџџџџџџџ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_namestates

Й
&model_2_simple_rnn_5_while_cond_240107F
Bmodel_2_simple_rnn_5_while_model_2_simple_rnn_5_while_loop_counterL
Hmodel_2_simple_rnn_5_while_model_2_simple_rnn_5_while_maximum_iterations*
&model_2_simple_rnn_5_while_placeholder,
(model_2_simple_rnn_5_while_placeholder_1,
(model_2_simple_rnn_5_while_placeholder_2H
Dmodel_2_simple_rnn_5_while_less_model_2_simple_rnn_5_strided_slice_1^
Zmodel_2_simple_rnn_5_while_model_2_simple_rnn_5_while_cond_240107___redundant_placeholder0^
Zmodel_2_simple_rnn_5_while_model_2_simple_rnn_5_while_cond_240107___redundant_placeholder1^
Zmodel_2_simple_rnn_5_while_model_2_simple_rnn_5_while_cond_240107___redundant_placeholder2^
Zmodel_2_simple_rnn_5_while_model_2_simple_rnn_5_while_cond_240107___redundant_placeholder3'
#model_2_simple_rnn_5_while_identity
Ж
model_2/simple_rnn_5/while/LessLess&model_2_simple_rnn_5_while_placeholderDmodel_2_simple_rnn_5_while_less_model_2_simple_rnn_5_strided_slice_1*
T0*
_output_shapes
: u
#model_2/simple_rnn_5/while/IdentityIdentity#model_2/simple_rnn_5/while/Less:z:0*
T0
*
_output_shapes
: "S
#model_2_simple_rnn_5_while_identity,model_2/simple_rnn_5/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :џџџџџџџџџ
: ::::: 
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
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
:

Й
&model_2_simple_rnn_6_while_cond_240216F
Bmodel_2_simple_rnn_6_while_model_2_simple_rnn_6_while_loop_counterL
Hmodel_2_simple_rnn_6_while_model_2_simple_rnn_6_while_maximum_iterations*
&model_2_simple_rnn_6_while_placeholder,
(model_2_simple_rnn_6_while_placeholder_1,
(model_2_simple_rnn_6_while_placeholder_2H
Dmodel_2_simple_rnn_6_while_less_model_2_simple_rnn_6_strided_slice_1^
Zmodel_2_simple_rnn_6_while_model_2_simple_rnn_6_while_cond_240216___redundant_placeholder0^
Zmodel_2_simple_rnn_6_while_model_2_simple_rnn_6_while_cond_240216___redundant_placeholder1^
Zmodel_2_simple_rnn_6_while_model_2_simple_rnn_6_while_cond_240216___redundant_placeholder2^
Zmodel_2_simple_rnn_6_while_model_2_simple_rnn_6_while_cond_240216___redundant_placeholder3'
#model_2_simple_rnn_6_while_identity
Ж
model_2/simple_rnn_6/while/LessLess&model_2_simple_rnn_6_while_placeholderDmodel_2_simple_rnn_6_while_less_model_2_simple_rnn_6_strided_slice_1*
T0*
_output_shapes
: u
#model_2/simple_rnn_6/while/IdentityIdentity#model_2/simple_rnn_6/while/Less:z:0*
T0
*
_output_shapes
: "S
#model_2_simple_rnn_6_while_identity,model_2/simple_rnn_6/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :џџџџџџџџџ: ::::: 
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:

ш
M__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_238598

inputs

states0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
2
 matmul_1_readvariableop_resource:


identity

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ
G
ReluReluadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
c

Identity_1IdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ

NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ:џџџџџџџџџ
: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ

 
_user_specified_namestates

Р
C__inference_model_2_layer_call_and_return_conditional_losses_240805

inputsO
=simple_rnn_5_simple_rnn_cell_5_matmul_readvariableop_resource:
L
>simple_rnn_5_simple_rnn_cell_5_biasadd_readvariableop_resource:
Q
?simple_rnn_5_simple_rnn_cell_5_matmul_1_readvariableop_resource:

O
=simple_rnn_6_simple_rnn_cell_6_matmul_readvariableop_resource:
L
>simple_rnn_6_simple_rnn_cell_6_biasadd_readvariableop_resource:Q
?simple_rnn_6_simple_rnn_cell_6_matmul_1_readvariableop_resource:
identityЂ5simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOpЂ4simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOpЂ6simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOpЂsimple_rnn_5/whileЂ5simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOpЂ4simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOpЂ6simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOpЂsimple_rnn_6/whileH
simple_rnn_5/ShapeShapeinputs*
T0*
_output_shapes
:j
 simple_rnn_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"simple_rnn_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"simple_rnn_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
simple_rnn_5/strided_sliceStridedSlicesimple_rnn_5/Shape:output:0)simple_rnn_5/strided_slice/stack:output:0+simple_rnn_5/strided_slice/stack_1:output:0+simple_rnn_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
simple_rnn_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :

simple_rnn_5/zeros/packedPack#simple_rnn_5/strided_slice:output:0$simple_rnn_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
simple_rnn_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
simple_rnn_5/zerosFill"simple_rnn_5/zeros/packed:output:0!simple_rnn_5/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
p
simple_rnn_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
simple_rnn_5/transpose	Transposeinputs$simple_rnn_5/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ^
simple_rnn_5/Shape_1Shapesimple_rnn_5/transpose:y:0*
T0*
_output_shapes
:l
"simple_rnn_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
simple_rnn_5/strided_slice_1StridedSlicesimple_rnn_5/Shape_1:output:0+simple_rnn_5/strided_slice_1/stack:output:0-simple_rnn_5/strided_slice_1/stack_1:output:0-simple_rnn_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(simple_rnn_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџл
simple_rnn_5/TensorArrayV2TensorListReserve1simple_rnn_5/TensorArrayV2/element_shape:output:0%simple_rnn_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
Bsimple_rnn_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
4simple_rnn_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_5/transpose:y:0Ksimple_rnn_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвl
"simple_rnn_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Њ
simple_rnn_5/strided_slice_2StridedSlicesimple_rnn_5/transpose:y:0+simple_rnn_5/strided_slice_2/stack:output:0-simple_rnn_5/strided_slice_2/stack_1:output:0-simple_rnn_5/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskВ
4simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp=simple_rnn_5_simple_rnn_cell_5_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0Ц
%simple_rnn_5/simple_rnn_cell_5/MatMulMatMul%simple_rnn_5/strided_slice_2:output:0<simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
А
5simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_5_simple_rnn_cell_5_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0г
&simple_rnn_5/simple_rnn_cell_5/BiasAddBiasAdd/simple_rnn_5/simple_rnn_cell_5/MatMul:product:0=simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Ж
6simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_5_simple_rnn_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype0Р
'simple_rnn_5/simple_rnn_cell_5/MatMul_1MatMulsimple_rnn_5/zeros:output:0>simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
С
"simple_rnn_5/simple_rnn_cell_5/addAddV2/simple_rnn_5/simple_rnn_cell_5/BiasAdd:output:01simple_rnn_5/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ

#simple_rnn_5/simple_rnn_cell_5/ReluRelu&simple_rnn_5/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
{
*simple_rnn_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   k
)simple_rnn_5/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :ь
simple_rnn_5/TensorArrayV2_1TensorListReserve3simple_rnn_5/TensorArrayV2_1/element_shape:output:02simple_rnn_5/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвS
simple_rnn_5/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%simple_rnn_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџa
simple_rnn_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
simple_rnn_5/whileWhile(simple_rnn_5/while/loop_counter:output:0.simple_rnn_5/while/maximum_iterations:output:0simple_rnn_5/time:output:0%simple_rnn_5/TensorArrayV2_1:handle:0simple_rnn_5/zeros:output:0%simple_rnn_5/strided_slice_1:output:0Dsimple_rnn_5/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_5_simple_rnn_cell_5_matmul_readvariableop_resource>simple_rnn_5_simple_rnn_cell_5_biasadd_readvariableop_resource?simple_rnn_5_simple_rnn_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( **
body"R 
simple_rnn_5_while_body_240630**
cond"R 
simple_rnn_5_while_cond_240629*8
output_shapes'
%: : : : :џџџџџџџџџ
: : : : : *
parallel_iterations 
=simple_rnn_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   §
/simple_rnn_5/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_5/while:output:3Fsimple_rnn_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ
*
element_dtype0*
num_elementsu
"simple_rnn_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџn
$simple_rnn_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ш
simple_rnn_5/strided_slice_3StridedSlice8simple_rnn_5/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_5/strided_slice_3/stack:output:0-simple_rnn_5/strided_slice_3/stack_1:output:0-simple_rnn_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ
*
shrink_axis_maskr
simple_rnn_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Н
simple_rnn_5/transpose_1	Transpose8simple_rnn_5/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_5/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
`
repeat_vector_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Ў
repeat_vector_2/ExpandDims
ExpandDims%simple_rnn_5/strided_slice_3:output:0'repeat_vector_2/ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
j
repeat_vector_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"         
repeat_vector_2/TileTile#repeat_vector_2/ExpandDims:output:0repeat_vector_2/stack:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
_
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
value	B :
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
:џџџџџџџџџp
simple_rnn_6/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
simple_rnn_6/transpose	Transposerepeat_vector_2/Tile:output:0$simple_rnn_6/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
^
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
valueB"џџџџ
   
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
valueB:Њ
simple_rnn_6/strided_slice_2StridedSlicesimple_rnn_6/transpose:y:0+simple_rnn_6/strided_slice_2/stack:output:0-simple_rnn_6/strided_slice_2/stack_1:output:0-simple_rnn_6/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ
*
shrink_axis_maskВ
4simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOpReadVariableOp=simple_rnn_6_simple_rnn_cell_6_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0Ц
%simple_rnn_6/simple_rnn_cell_6/MatMulMatMul%simple_rnn_6/strided_slice_2:output:0<simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџА
5simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_6_simple_rnn_cell_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0г
&simple_rnn_6/simple_rnn_cell_6/BiasAddBiasAdd/simple_rnn_6/simple_rnn_cell_6/MatMul:product:0=simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЖ
6simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_6_simple_rnn_cell_6_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0Р
'simple_rnn_6/simple_rnn_cell_6/MatMul_1MatMulsimple_rnn_6/zeros:output:0>simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџС
"simple_rnn_6/simple_rnn_cell_6/addAddV2/simple_rnn_6/simple_rnn_cell_6/BiasAdd:output:01simple_rnn_6/simple_rnn_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ
#simple_rnn_6/simple_rnn_cell_6/ReluRelu&simple_rnn_6/simple_rnn_cell_6/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ{
*simple_rnn_6/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   п
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
value	B : 
simple_rnn_6/whileWhile(simple_rnn_6/while/loop_counter:output:0.simple_rnn_6/while/maximum_iterations:output:0simple_rnn_6/time:output:0%simple_rnn_6/TensorArrayV2_1:handle:0simple_rnn_6/zeros:output:0%simple_rnn_6/strided_slice_1:output:0Dsimple_rnn_6/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_6_simple_rnn_cell_6_matmul_readvariableop_resource>simple_rnn_6_simple_rnn_cell_6_biasadd_readvariableop_resource?simple_rnn_6_simple_rnn_cell_6_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( **
body"R 
simple_rnn_6_while_body_240739**
cond"R 
simple_rnn_6_while_cond_240738*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 
=simple_rnn_6/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   щ
/simple_rnn_6/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_6/while:output:3Fsimple_rnn_6/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
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
:џџџџџџџџџ*
shrink_axis_maskr
simple_rnn_6/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Н
simple_rnn_6/transpose_1	Transpose8simple_rnn_6/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_6/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџo
IdentityIdentitysimple_rnn_6/transpose_1:y:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџР
NoOpNoOp6^simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp5^simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp7^simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp^simple_rnn_5/while6^simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOp5^simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOp7^simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOp^simple_rnn_6/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : : : : : 2n
5simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp5simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp2l
4simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp4simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp2p
6simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp6simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp2(
simple_rnn_5/whilesimple_rnn_5/while2n
5simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOp5simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOp2l
4simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOp4simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOp2p
6simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOp6simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOp2(
simple_rnn_6/whilesimple_rnn_6/while:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ј4

H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_238676

inputs*
simple_rnn_cell_5_238599:
&
simple_rnn_cell_5_238601:
*
simple_rnn_cell_5_238603:


identityЂ)simple_rnn_cell_5/StatefulPartitionedCallЂwhile;
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
value	B :
s
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
:џџџџџџџџџ
c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџD
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
valueB"џџџџ   р
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskч
)simple_rnn_cell_5/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_5_238599simple_rnn_cell_5_238601simple_rnn_cell_5_238603*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ
:џџџџџџџџџ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_238598n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   ^
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
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_5_238599simple_rnn_cell_5_238601simple_rnn_cell_5_238603*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_238612*
condR
while_cond_238611*8
output_shapes'
%: : : : :џџџџџџџџџ
: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ
*
element_dtype0*
num_elementsh
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
:џџџџџџџџџ
*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
z
NoOpNoOp*^simple_rnn_cell_5/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2V
)simple_rnn_cell_5/StatefulPartitionedCall)simple_rnn_cell_5/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
к
Њ
while_cond_241478
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_241478___redundant_placeholder04
0while_while_cond_241478___redundant_placeholder14
0while_while_cond_241478___redundant_placeholder24
0while_while_cond_241478___redundant_placeholder3
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
-: : : : :џџџџџџџџџ
: ::::: 
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
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
:
@
П
&model_2_simple_rnn_6_while_body_240461F
Bmodel_2_simple_rnn_6_while_model_2_simple_rnn_6_while_loop_counterL
Hmodel_2_simple_rnn_6_while_model_2_simple_rnn_6_while_maximum_iterations*
&model_2_simple_rnn_6_while_placeholder,
(model_2_simple_rnn_6_while_placeholder_1,
(model_2_simple_rnn_6_while_placeholder_2E
Amodel_2_simple_rnn_6_while_model_2_simple_rnn_6_strided_slice_1_0
}model_2_simple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_model_2_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor_0_
Mmodel_2_simple_rnn_6_while_simple_rnn_cell_6_matmul_readvariableop_resource_0:
\
Nmodel_2_simple_rnn_6_while_simple_rnn_cell_6_biasadd_readvariableop_resource_0:a
Omodel_2_simple_rnn_6_while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0:'
#model_2_simple_rnn_6_while_identity)
%model_2_simple_rnn_6_while_identity_1)
%model_2_simple_rnn_6_while_identity_2)
%model_2_simple_rnn_6_while_identity_3)
%model_2_simple_rnn_6_while_identity_4C
?model_2_simple_rnn_6_while_model_2_simple_rnn_6_strided_slice_1
{model_2_simple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_model_2_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor]
Kmodel_2_simple_rnn_6_while_simple_rnn_cell_6_matmul_readvariableop_resource:
Z
Lmodel_2_simple_rnn_6_while_simple_rnn_cell_6_biasadd_readvariableop_resource:_
Mmodel_2_simple_rnn_6_while_simple_rnn_cell_6_matmul_1_readvariableop_resource:ЂCmodel_2/simple_rnn_6/while/simple_rnn_cell_6/BiasAdd/ReadVariableOpЂBmodel_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul/ReadVariableOpЂDmodel_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp
Lmodel_2/simple_rnn_6/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   
>model_2/simple_rnn_6/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}model_2_simple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_model_2_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor_0&model_2_simple_rnn_6_while_placeholderUmodel_2/simple_rnn_6/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ
*
element_dtype0а
Bmodel_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul/ReadVariableOpReadVariableOpMmodel_2_simple_rnn_6_while_simple_rnn_cell_6_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype0
3model_2/simple_rnn_6/while/simple_rnn_cell_6/MatMulMatMulEmodel_2/simple_rnn_6/while/TensorArrayV2Read/TensorListGetItem:item:0Jmodel_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЮ
Cmodel_2/simple_rnn_6/while/simple_rnn_cell_6/BiasAdd/ReadVariableOpReadVariableOpNmodel_2_simple_rnn_6_while_simple_rnn_cell_6_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0§
4model_2/simple_rnn_6/while/simple_rnn_cell_6/BiasAddBiasAdd=model_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul:product:0Kmodel_2/simple_rnn_6/while/simple_rnn_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџд
Dmodel_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul_1/ReadVariableOpReadVariableOpOmodel_2_simple_rnn_6_while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0щ
5model_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul_1MatMul(model_2_simple_rnn_6_while_placeholder_2Lmodel_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџы
0model_2/simple_rnn_6/while/simple_rnn_cell_6/addAddV2=model_2/simple_rnn_6/while/simple_rnn_cell_6/BiasAdd:output:0?model_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџЁ
1model_2/simple_rnn_6/while/simple_rnn_cell_6/ReluRelu4model_2/simple_rnn_6/while/simple_rnn_cell_6/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџЇ
?model_2/simple_rnn_6/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(model_2_simple_rnn_6_while_placeholder_1&model_2_simple_rnn_6_while_placeholder?model_2/simple_rnn_6/while/simple_rnn_cell_6/Relu:activations:0*
_output_shapes
: *
element_dtype0:щшвb
 model_2/simple_rnn_6/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
model_2/simple_rnn_6/while/addAddV2&model_2_simple_rnn_6_while_placeholder)model_2/simple_rnn_6/while/add/y:output:0*
T0*
_output_shapes
: d
"model_2/simple_rnn_6/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Л
 model_2/simple_rnn_6/while/add_1AddV2Bmodel_2_simple_rnn_6_while_model_2_simple_rnn_6_while_loop_counter+model_2/simple_rnn_6/while/add_1/y:output:0*
T0*
_output_shapes
: 
#model_2/simple_rnn_6/while/IdentityIdentity$model_2/simple_rnn_6/while/add_1:z:0 ^model_2/simple_rnn_6/while/NoOp*
T0*
_output_shapes
: О
%model_2/simple_rnn_6/while/Identity_1IdentityHmodel_2_simple_rnn_6_while_model_2_simple_rnn_6_while_maximum_iterations ^model_2/simple_rnn_6/while/NoOp*
T0*
_output_shapes
: 
%model_2/simple_rnn_6/while/Identity_2Identity"model_2/simple_rnn_6/while/add:z:0 ^model_2/simple_rnn_6/while/NoOp*
T0*
_output_shapes
: Х
%model_2/simple_rnn_6/while/Identity_3IdentityOmodel_2/simple_rnn_6/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^model_2/simple_rnn_6/while/NoOp*
T0*
_output_shapes
: Ц
%model_2/simple_rnn_6/while/Identity_4Identity?model_2/simple_rnn_6/while/simple_rnn_cell_6/Relu:activations:0 ^model_2/simple_rnn_6/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџГ
model_2/simple_rnn_6/while/NoOpNoOpD^model_2/simple_rnn_6/while/simple_rnn_cell_6/BiasAdd/ReadVariableOpC^model_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul/ReadVariableOpE^model_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "S
#model_2_simple_rnn_6_while_identity,model_2/simple_rnn_6/while/Identity:output:0"W
%model_2_simple_rnn_6_while_identity_1.model_2/simple_rnn_6/while/Identity_1:output:0"W
%model_2_simple_rnn_6_while_identity_2.model_2/simple_rnn_6/while/Identity_2:output:0"W
%model_2_simple_rnn_6_while_identity_3.model_2/simple_rnn_6/while/Identity_3:output:0"W
%model_2_simple_rnn_6_while_identity_4.model_2/simple_rnn_6/while/Identity_4:output:0"
?model_2_simple_rnn_6_while_model_2_simple_rnn_6_strided_slice_1Amodel_2_simple_rnn_6_while_model_2_simple_rnn_6_strided_slice_1_0"
Lmodel_2_simple_rnn_6_while_simple_rnn_cell_6_biasadd_readvariableop_resourceNmodel_2_simple_rnn_6_while_simple_rnn_cell_6_biasadd_readvariableop_resource_0" 
Mmodel_2_simple_rnn_6_while_simple_rnn_cell_6_matmul_1_readvariableop_resourceOmodel_2_simple_rnn_6_while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0"
Kmodel_2_simple_rnn_6_while_simple_rnn_cell_6_matmul_readvariableop_resourceMmodel_2_simple_rnn_6_while_simple_rnn_cell_6_matmul_readvariableop_resource_0"ќ
{model_2_simple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_model_2_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor}model_2_simple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_model_2_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ: : : : : 2
Cmodel_2/simple_rnn_6/while/simple_rnn_cell_6/BiasAdd/ReadVariableOpCmodel_2/simple_rnn_6/while/simple_rnn_cell_6/BiasAdd/ReadVariableOp2
Bmodel_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul/ReadVariableOpBmodel_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul/ReadVariableOp2
Dmodel_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul_1/ReadVariableOpDmodel_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
і 
б
while_body_239081
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
 while_simple_rnn_cell_6_239103_0:
.
 while_simple_rnn_cell_6_239105_0:2
 while_simple_rnn_cell_6_239107_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
while_simple_rnn_cell_6_239103:
,
while_simple_rnn_cell_6_239105:0
while_simple_rnn_cell_6_239107:Ђ/while/simple_rnn_cell_6/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ
*
element_dtype0Ђ
/while/simple_rnn_cell_6/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2 while_simple_rnn_cell_6_239103_0 while_simple_rnn_cell_6_239105_0 while_simple_rnn_cell_6_239107_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_simple_rnn_cell_6_layer_call_and_return_conditional_losses_239029с
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
:џџџџџџџџџ~

while/NoOpNoOp0^while/simple_rnn_cell_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"B
while_simple_rnn_cell_6_239103 while_simple_rnn_cell_6_239103_0"B
while_simple_rnn_cell_6_239105 while_simple_rnn_cell_6_239105_0"B
while_simple_rnn_cell_6_239107 while_simple_rnn_cell_6_239107_0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ: : : : : 2b
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 

Й
-__inference_simple_rnn_5_layer_call_fn_241073
inputs_0
unknown:

	unknown_0:

	unknown_1:


identityЂStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_238676o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
К

Ё
simple_rnn_6_while_cond_2407386
2simple_rnn_6_while_simple_rnn_6_while_loop_counter<
8simple_rnn_6_while_simple_rnn_6_while_maximum_iterations"
simple_rnn_6_while_placeholder$
 simple_rnn_6_while_placeholder_1$
 simple_rnn_6_while_placeholder_28
4simple_rnn_6_while_less_simple_rnn_6_strided_slice_1N
Jsimple_rnn_6_while_simple_rnn_6_while_cond_240738___redundant_placeholder0N
Jsimple_rnn_6_while_simple_rnn_6_while_cond_240738___redundant_placeholder1N
Jsimple_rnn_6_while_simple_rnn_6_while_cond_240738___redundant_placeholder2N
Jsimple_rnn_6_while_simple_rnn_6_while_cond_240738___redundant_placeholder3
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
-: : : : :џџџџџџџџџ: ::::: 
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
Ј>
Л
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_239269

inputsB
0simple_rnn_cell_5_matmul_readvariableop_resource:
?
1simple_rnn_cell_5_biasadd_readvariableop_resource:
D
2simple_rnn_cell_5_matmul_1_readvariableop_resource:


identityЂ(simple_rnn_cell_5/BiasAdd/ReadVariableOpЂ'simple_rnn_cell_5/MatMul/ReadVariableOpЂ)simple_rnn_cell_5/MatMul_1/ReadVariableOpЂwhile;
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
value	B :
s
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
:џџџџџџџџџ
c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџD
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
valueB"џџџџ   р
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
'simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_5_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
simple_rnn_cell_5/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ

(simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_5_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Ќ
simple_rnn_cell_5/BiasAddBiasAdd"simple_rnn_cell_5/MatMul:product:00simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ

)simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype0
simple_rnn_cell_5/MatMul_1MatMulzeros:output:01simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ

simple_rnn_cell_5/addAddV2"simple_rnn_cell_5/BiasAdd:output:0$simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ
k
simple_rnn_cell_5/ReluRelusimple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   ^
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
value	B : и
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_5_matmul_readvariableop_resource1simple_rnn_cell_5_biasadd_readvariableop_resource2simple_rnn_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_239202*
condR
while_cond_239201*8
output_shapes'
%: : : : :џџџџџџџџџ
: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ
*
element_dtype0*
num_elementsh
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
:џџџџџџџџџ
*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
Я
NoOpNoOp)^simple_rnn_cell_5/BiasAdd/ReadVariableOp(^simple_rnn_cell_5/MatMul/ReadVariableOp*^simple_rnn_cell_5/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2T
(simple_rnn_cell_5/BiasAdd/ReadVariableOp(simple_rnn_cell_5/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_5/MatMul/ReadVariableOp'simple_rnn_cell_5/MatMul/ReadVariableOp2V
)simple_rnn_cell_5/MatMul_1/ReadVariableOp)simple_rnn_cell_5/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
4

H__inference_simple_rnn_6_layer_call_and_return_conditional_losses_238985

inputs*
simple_rnn_cell_6_238910:
&
simple_rnn_cell_6_238912:*
simple_rnn_cell_6_238914:
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
value	B :s
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
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
D
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
valueB"џџџџ
   р
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ
*
shrink_axis_maskч
)simple_rnn_cell_6/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_6_238910simple_rnn_cell_6_238912simple_rnn_cell_6_238914*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_simple_rnn_cell_6_layer_call_and_return_conditional_losses_238909n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
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
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_6_238910simple_rnn_cell_6_238912simple_rnn_cell_6_238914*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_238922*
condR
while_cond_238921*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
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
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџk
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџz
NoOpNoOp*^simple_rnn_cell_6/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ
: : : 2V
)simple_rnn_cell_6/StatefulPartitionedCall)simple_rnn_cell_6/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ

 
_user_specified_nameinputs
П,
Ш
while_body_241645
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0J
8while_simple_rnn_cell_6_matmul_readvariableop_resource_0:
G
9while_simple_rnn_cell_6_biasadd_readvariableop_resource_0:L
:while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorH
6while_simple_rnn_cell_6_matmul_readvariableop_resource:
E
7while_simple_rnn_cell_6_biasadd_readvariableop_resource:J
8while_simple_rnn_cell_6_matmul_1_readvariableop_resource:Ђ.while/simple_rnn_cell_6/BiasAdd/ReadVariableOpЂ-while/simple_rnn_cell_6/MatMul/ReadVariableOpЂ/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   І
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ
*
element_dtype0І
-while/simple_rnn_cell_6/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_6_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype0У
while/simple_rnn_cell_6/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЄ
.while/simple_rnn_cell_6/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_6_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0О
while/simple_rnn_cell_6/BiasAddBiasAdd(while/simple_rnn_cell_6/MatMul:product:06while/simple_rnn_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЊ
/while/simple_rnn_cell_6/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0Њ
 while/simple_rnn_cell_6/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЌ
while/simple_rnn_cell_6/addAddV2(while/simple_rnn_cell_6/BiasAdd:output:0*while/simple_rnn_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџw
while/simple_rnn_cell_6/ReluReluwhile/simple_rnn_cell_6/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџг
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder*while/simple_rnn_cell_6/Relu:activations:0*
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
: 
while/Identity_4Identity*while/simple_rnn_cell_6/Relu:activations:0^while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџп

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
%: : : : :џџџџџџџџџ: : : : : 2`
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
к	
Р
-__inference_sequential_5_layer_call_fn_240044

inputs
unknown:

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identityЂStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_239836s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
к
Њ
while_cond_241644
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_241644___redundant_placeholder04
0while_while_cond_241644___redundant_placeholder14
0while_while_cond_241644___redundant_placeholder24
0while_while_cond_241644___redundant_placeholder3
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
-: : : : :џџџџџџџџџ: ::::: 
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
к
Њ
while_cond_241148
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_241148___redundant_placeholder04
0while_while_cond_241148___redundant_placeholder14
0while_while_cond_241148___redundant_placeholder24
0while_while_cond_241148___redundant_placeholder3
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
-: : : : :џџџџџџџџџ
: ::::: 
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
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
:
Й

к
2__inference_simple_rnn_cell_6_layer_call_fn_242125

inputs
states_0
unknown:

	unknown_0:
	unknown_1:
identity

identity_1ЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_simple_rnn_cell_6_layer_call_and_return_conditional_losses_239029o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ
:џџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0
Ы>
Н
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_241216
inputs_0B
0simple_rnn_cell_5_matmul_readvariableop_resource:
?
1simple_rnn_cell_5_biasadd_readvariableop_resource:
D
2simple_rnn_cell_5_matmul_1_readvariableop_resource:


identityЂ(simple_rnn_cell_5/BiasAdd/ReadVariableOpЂ'simple_rnn_cell_5/MatMul/ReadVariableOpЂ)simple_rnn_cell_5/MatMul_1/ReadVariableOpЂwhile=
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
value	B :
s
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
:џџџџџџџџџ
c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџD
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
valueB"џџџџ   р
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
'simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_5_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
simple_rnn_cell_5/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ

(simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_5_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Ќ
simple_rnn_cell_5/BiasAddBiasAdd"simple_rnn_cell_5/MatMul:product:00simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ

)simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype0
simple_rnn_cell_5/MatMul_1MatMulzeros:output:01simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ

simple_rnn_cell_5/addAddV2"simple_rnn_cell_5/BiasAdd:output:0$simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ
k
simple_rnn_cell_5/ReluRelusimple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   ^
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
value	B : и
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_5_matmul_readvariableop_resource1simple_rnn_cell_5_biasadd_readvariableop_resource2simple_rnn_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_241149*
condR
while_cond_241148*8
output_shapes'
%: : : : :џџџџџџџџџ
: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ
*
element_dtype0*
num_elementsh
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
:џџџџџџџџџ
*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
Я
NoOpNoOp)^simple_rnn_cell_5/BiasAdd/ReadVariableOp(^simple_rnn_cell_5/MatMul/ReadVariableOp*^simple_rnn_cell_5/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2T
(simple_rnn_cell_5/BiasAdd/ReadVariableOp(simple_rnn_cell_5/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_5/MatMul/ReadVariableOp'simple_rnn_cell_5/MatMul/ReadVariableOp2V
)simple_rnn_cell_5/MatMul_1/ReadVariableOp)simple_rnn_cell_5/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0

ш
M__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_238720

inputs

states0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
2
 matmul_1_readvariableop_resource:


identity

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ
G
ReluReluadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
c

Identity_1IdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ

NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ:џџџџџџџџџ
: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ

 
_user_specified_namestates

Й
&model_2_simple_rnn_6_while_cond_240460F
Bmodel_2_simple_rnn_6_while_model_2_simple_rnn_6_while_loop_counterL
Hmodel_2_simple_rnn_6_while_model_2_simple_rnn_6_while_maximum_iterations*
&model_2_simple_rnn_6_while_placeholder,
(model_2_simple_rnn_6_while_placeholder_1,
(model_2_simple_rnn_6_while_placeholder_2H
Dmodel_2_simple_rnn_6_while_less_model_2_simple_rnn_6_strided_slice_1^
Zmodel_2_simple_rnn_6_while_model_2_simple_rnn_6_while_cond_240460___redundant_placeholder0^
Zmodel_2_simple_rnn_6_while_model_2_simple_rnn_6_while_cond_240460___redundant_placeholder1^
Zmodel_2_simple_rnn_6_while_model_2_simple_rnn_6_while_cond_240460___redundant_placeholder2^
Zmodel_2_simple_rnn_6_while_model_2_simple_rnn_6_while_cond_240460___redundant_placeholder3'
#model_2_simple_rnn_6_while_identity
Ж
model_2/simple_rnn_6/while/LessLess&model_2_simple_rnn_6_while_placeholderDmodel_2_simple_rnn_6_while_less_model_2_simple_rnn_6_strided_slice_1*
T0*
_output_shapes
: u
#model_2/simple_rnn_6/while/IdentityIdentity#model_2/simple_rnn_6/while/Less:z:0*
T0
*
_output_shapes
: "S
#model_2_simple_rnn_6_while_identity,model_2/simple_rnn_6/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :џџџџџџџџџ: ::::: 
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
к
Њ
while_cond_239594
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_239594___redundant_placeholder04
0while_while_cond_239594___redundant_placeholder14
0while_while_cond_239594___redundant_placeholder24
0while_while_cond_239594___redundant_placeholder3
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
-: : : : :џџџџџџџџџ
: ::::: 
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
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
:

Й
&model_2_simple_rnn_5_while_cond_240351F
Bmodel_2_simple_rnn_5_while_model_2_simple_rnn_5_while_loop_counterL
Hmodel_2_simple_rnn_5_while_model_2_simple_rnn_5_while_maximum_iterations*
&model_2_simple_rnn_5_while_placeholder,
(model_2_simple_rnn_5_while_placeholder_1,
(model_2_simple_rnn_5_while_placeholder_2H
Dmodel_2_simple_rnn_5_while_less_model_2_simple_rnn_5_strided_slice_1^
Zmodel_2_simple_rnn_5_while_model_2_simple_rnn_5_while_cond_240351___redundant_placeholder0^
Zmodel_2_simple_rnn_5_while_model_2_simple_rnn_5_while_cond_240351___redundant_placeholder1^
Zmodel_2_simple_rnn_5_while_model_2_simple_rnn_5_while_cond_240351___redundant_placeholder2^
Zmodel_2_simple_rnn_5_while_model_2_simple_rnn_5_while_cond_240351___redundant_placeholder3'
#model_2_simple_rnn_5_while_identity
Ж
model_2/simple_rnn_5/while/LessLess&model_2_simple_rnn_5_while_placeholderDmodel_2_simple_rnn_5_while_less_model_2_simple_rnn_5_strided_slice_1*
T0*
_output_shapes
: u
#model_2/simple_rnn_5/while/IdentityIdentity#model_2/simple_rnn_5/while/Less:z:0*
T0
*
_output_shapes
: "S
#model_2_simple_rnn_5_while_identity,model_2/simple_rnn_5/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :џџџџџџџџџ
: ::::: 
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
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
:
ќ9
Ю
simple_rnn_5_while_body_2408486
2simple_rnn_5_while_simple_rnn_5_while_loop_counter<
8simple_rnn_5_while_simple_rnn_5_while_maximum_iterations"
simple_rnn_5_while_placeholder$
 simple_rnn_5_while_placeholder_1$
 simple_rnn_5_while_placeholder_25
1simple_rnn_5_while_simple_rnn_5_strided_slice_1_0q
msimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0W
Esimple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resource_0:
T
Fsimple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resource_0:
Y
Gsimple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0:


simple_rnn_5_while_identity!
simple_rnn_5_while_identity_1!
simple_rnn_5_while_identity_2!
simple_rnn_5_while_identity_3!
simple_rnn_5_while_identity_43
/simple_rnn_5_while_simple_rnn_5_strided_slice_1o
ksimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensorU
Csimple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resource:
R
Dsimple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resource:
W
Esimple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource:

Ђ;simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOpЂ:simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOpЂ<simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp
Dsimple_rnn_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ч
6simple_rnn_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_5_while_placeholderMsimple_rnn_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0Р
:simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype0ъ
+simple_rnn_5/while/simple_rnn_cell_5/MatMulMatMul=simple_rnn_5/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
О
;simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resource_0*
_output_shapes
:
*
dtype0х
,simple_rnn_5/while/simple_rnn_cell_5/BiasAddBiasAdd5simple_rnn_5/while/simple_rnn_cell_5/MatMul:product:0Csimple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Ф
<simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:

*
dtype0б
-simple_rnn_5/while/simple_rnn_cell_5/MatMul_1MatMul simple_rnn_5_while_placeholder_2Dsimple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
г
(simple_rnn_5/while/simple_rnn_cell_5/addAddV25simple_rnn_5/while/simple_rnn_cell_5/BiasAdd:output:07simple_rnn_5/while/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ

)simple_rnn_5/while/simple_rnn_cell_5/ReluRelu,simple_rnn_5/while/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ

=simple_rnn_5/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Џ
7simple_rnn_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_5_while_placeholder_1Fsimple_rnn_5/while/TensorArrayV2Write/TensorListSetItem/index:output:07simple_rnn_5/while/simple_rnn_cell_5/Relu:activations:0*
_output_shapes
: *
element_dtype0:щшвZ
simple_rnn_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
simple_rnn_5/while/addAddV2simple_rnn_5_while_placeholder!simple_rnn_5/while/add/y:output:0*
T0*
_output_shapes
: \
simple_rnn_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
simple_rnn_5/while/add_1AddV22simple_rnn_5_while_simple_rnn_5_while_loop_counter#simple_rnn_5/while/add_1/y:output:0*
T0*
_output_shapes
: 
simple_rnn_5/while/IdentityIdentitysimple_rnn_5/while/add_1:z:0^simple_rnn_5/while/NoOp*
T0*
_output_shapes
: 
simple_rnn_5/while/Identity_1Identity8simple_rnn_5_while_simple_rnn_5_while_maximum_iterations^simple_rnn_5/while/NoOp*
T0*
_output_shapes
: 
simple_rnn_5/while/Identity_2Identitysimple_rnn_5/while/add:z:0^simple_rnn_5/while/NoOp*
T0*
_output_shapes
: ­
simple_rnn_5/while/Identity_3IdentityGsimple_rnn_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_5/while/NoOp*
T0*
_output_shapes
: Ў
simple_rnn_5/while/Identity_4Identity7simple_rnn_5/while/simple_rnn_cell_5/Relu:activations:0^simple_rnn_5/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ

simple_rnn_5/while/NoOpNoOp<^simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp;^simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp=^simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "C
simple_rnn_5_while_identity$simple_rnn_5/while/Identity:output:0"G
simple_rnn_5_while_identity_1&simple_rnn_5/while/Identity_1:output:0"G
simple_rnn_5_while_identity_2&simple_rnn_5/while/Identity_2:output:0"G
simple_rnn_5_while_identity_3&simple_rnn_5/while/Identity_3:output:0"G
simple_rnn_5_while_identity_4&simple_rnn_5/while/Identity_4:output:0"d
/simple_rnn_5_while_simple_rnn_5_strided_slice_11simple_rnn_5_while_simple_rnn_5_strided_slice_1_0"
Dsimple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resourceFsimple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resource_0"
Esimple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resourceGsimple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0"
Csimple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resourceEsimple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resource_0"м
ksimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensormsimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ
: : : : : 2z
;simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp;simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp2x
:simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp:simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp2|
<simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp<simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
: 
г8
Ю
simple_rnn_6_while_body_2407396
2simple_rnn_6_while_simple_rnn_6_while_loop_counter<
8simple_rnn_6_while_simple_rnn_6_while_maximum_iterations"
simple_rnn_6_while_placeholder$
 simple_rnn_6_while_placeholder_1$
 simple_rnn_6_while_placeholder_25
1simple_rnn_6_while_simple_rnn_6_strided_slice_1_0q
msimple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor_0W
Esimple_rnn_6_while_simple_rnn_cell_6_matmul_readvariableop_resource_0:
T
Fsimple_rnn_6_while_simple_rnn_cell_6_biasadd_readvariableop_resource_0:Y
Gsimple_rnn_6_while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0:
simple_rnn_6_while_identity!
simple_rnn_6_while_identity_1!
simple_rnn_6_while_identity_2!
simple_rnn_6_while_identity_3!
simple_rnn_6_while_identity_43
/simple_rnn_6_while_simple_rnn_6_strided_slice_1o
ksimple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_6_tensorarrayunstack_tensorlistfromtensorU
Csimple_rnn_6_while_simple_rnn_cell_6_matmul_readvariableop_resource:
R
Dsimple_rnn_6_while_simple_rnn_cell_6_biasadd_readvariableop_resource:W
Esimple_rnn_6_while_simple_rnn_cell_6_matmul_1_readvariableop_resource:Ђ;simple_rnn_6/while/simple_rnn_cell_6/BiasAdd/ReadVariableOpЂ:simple_rnn_6/while/simple_rnn_cell_6/MatMul/ReadVariableOpЂ<simple_rnn_6/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp
Dsimple_rnn_6/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   ч
6simple_rnn_6/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_6_while_placeholderMsimple_rnn_6/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ
*
element_dtype0Р
:simple_rnn_6/while/simple_rnn_cell_6/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_6_while_simple_rnn_cell_6_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype0ъ
+simple_rnn_6/while/simple_rnn_cell_6/MatMulMatMul=simple_rnn_6/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_6/while/simple_rnn_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџО
;simple_rnn_6/while/simple_rnn_cell_6/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_6_while_simple_rnn_cell_6_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0х
,simple_rnn_6/while/simple_rnn_cell_6/BiasAddBiasAdd5simple_rnn_6/while/simple_rnn_cell_6/MatMul:product:0Csimple_rnn_6/while/simple_rnn_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџФ
<simple_rnn_6/while/simple_rnn_cell_6/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_6_while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0б
-simple_rnn_6/while/simple_rnn_cell_6/MatMul_1MatMul simple_rnn_6_while_placeholder_2Dsimple_rnn_6/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџг
(simple_rnn_6/while/simple_rnn_cell_6/addAddV25simple_rnn_6/while/simple_rnn_cell_6/BiasAdd:output:07simple_rnn_6/while/simple_rnn_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ
)simple_rnn_6/while/simple_rnn_cell_6/ReluRelu,simple_rnn_6/while/simple_rnn_cell_6/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
7simple_rnn_6/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_6_while_placeholder_1simple_rnn_6_while_placeholder7simple_rnn_6/while/simple_rnn_cell_6/Relu:activations:0*
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
: Ў
simple_rnn_6/while/Identity_4Identity7simple_rnn_6/while/simple_rnn_cell_6/Relu:activations:0^simple_rnn_6/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
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
%: : : : :џџџџџџџџџ: : : : : 2z
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
к
Њ
while_cond_241368
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_241368___redundant_placeholder04
0while_while_cond_241368___redundant_placeholder14
0while_while_cond_241368___redundant_placeholder24
0while_while_cond_241368___redundant_placeholder3
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
-: : : : :џџџџџџџџџ
: ::::: 
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
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
:
к
Њ
while_cond_238611
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_238611___redundant_placeholder04
0while_while_cond_238611___redundant_placeholder14
0while_while_cond_238611___redundant_placeholder24
0while_while_cond_238611___redundant_placeholder3
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
-: : : : :џџџџџџџџџ
: ::::: 
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
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
:
џ

(__inference_model_2_layer_call_fn_239741
input_6
unknown:

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:
	unknown_4:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_239709s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_6
А
Й
-__inference_simple_rnn_6_layer_call_fn_241570
inputs_0
unknown:

	unknown_0:
	unknown_1:
identityЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_simple_rnn_6_layer_call_and_return_conditional_losses_238985|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ

"
_user_specified_name
inputs/0
ТA
П
&model_2_simple_rnn_5_while_body_240352F
Bmodel_2_simple_rnn_5_while_model_2_simple_rnn_5_while_loop_counterL
Hmodel_2_simple_rnn_5_while_model_2_simple_rnn_5_while_maximum_iterations*
&model_2_simple_rnn_5_while_placeholder,
(model_2_simple_rnn_5_while_placeholder_1,
(model_2_simple_rnn_5_while_placeholder_2E
Amodel_2_simple_rnn_5_while_model_2_simple_rnn_5_strided_slice_1_0
}model_2_simple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_model_2_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0_
Mmodel_2_simple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resource_0:
\
Nmodel_2_simple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resource_0:
a
Omodel_2_simple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0:

'
#model_2_simple_rnn_5_while_identity)
%model_2_simple_rnn_5_while_identity_1)
%model_2_simple_rnn_5_while_identity_2)
%model_2_simple_rnn_5_while_identity_3)
%model_2_simple_rnn_5_while_identity_4C
?model_2_simple_rnn_5_while_model_2_simple_rnn_5_strided_slice_1
{model_2_simple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_model_2_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor]
Kmodel_2_simple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resource:
Z
Lmodel_2_simple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resource:
_
Mmodel_2_simple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource:

ЂCmodel_2/simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOpЂBmodel_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOpЂDmodel_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp
Lmodel_2/simple_rnn_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
>model_2/simple_rnn_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}model_2_simple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_model_2_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0&model_2_simple_rnn_5_while_placeholderUmodel_2/simple_rnn_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0а
Bmodel_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOpMmodel_2_simple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype0
3model_2/simple_rnn_5/while/simple_rnn_cell_5/MatMulMatMulEmodel_2/simple_rnn_5/while/TensorArrayV2Read/TensorListGetItem:item:0Jmodel_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Ю
Cmodel_2/simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOpNmodel_2_simple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resource_0*
_output_shapes
:
*
dtype0§
4model_2/simple_rnn_5/while/simple_rnn_cell_5/BiasAddBiasAdd=model_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul:product:0Kmodel_2/simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
д
Dmodel_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOpOmodel_2_simple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:

*
dtype0щ
5model_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1MatMul(model_2_simple_rnn_5_while_placeholder_2Lmodel_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
ы
0model_2/simple_rnn_5/while/simple_rnn_cell_5/addAddV2=model_2/simple_rnn_5/while/simple_rnn_cell_5/BiasAdd:output:0?model_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ
Ё
1model_2/simple_rnn_5/while/simple_rnn_cell_5/ReluRelu4model_2/simple_rnn_5/while/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ

Emodel_2/simple_rnn_5/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Я
?model_2/simple_rnn_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(model_2_simple_rnn_5_while_placeholder_1Nmodel_2/simple_rnn_5/while/TensorArrayV2Write/TensorListSetItem/index:output:0?model_2/simple_rnn_5/while/simple_rnn_cell_5/Relu:activations:0*
_output_shapes
: *
element_dtype0:щшвb
 model_2/simple_rnn_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
model_2/simple_rnn_5/while/addAddV2&model_2_simple_rnn_5_while_placeholder)model_2/simple_rnn_5/while/add/y:output:0*
T0*
_output_shapes
: d
"model_2/simple_rnn_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Л
 model_2/simple_rnn_5/while/add_1AddV2Bmodel_2_simple_rnn_5_while_model_2_simple_rnn_5_while_loop_counter+model_2/simple_rnn_5/while/add_1/y:output:0*
T0*
_output_shapes
: 
#model_2/simple_rnn_5/while/IdentityIdentity$model_2/simple_rnn_5/while/add_1:z:0 ^model_2/simple_rnn_5/while/NoOp*
T0*
_output_shapes
: О
%model_2/simple_rnn_5/while/Identity_1IdentityHmodel_2_simple_rnn_5_while_model_2_simple_rnn_5_while_maximum_iterations ^model_2/simple_rnn_5/while/NoOp*
T0*
_output_shapes
: 
%model_2/simple_rnn_5/while/Identity_2Identity"model_2/simple_rnn_5/while/add:z:0 ^model_2/simple_rnn_5/while/NoOp*
T0*
_output_shapes
: Х
%model_2/simple_rnn_5/while/Identity_3IdentityOmodel_2/simple_rnn_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^model_2/simple_rnn_5/while/NoOp*
T0*
_output_shapes
: Ц
%model_2/simple_rnn_5/while/Identity_4Identity?model_2/simple_rnn_5/while/simple_rnn_cell_5/Relu:activations:0 ^model_2/simple_rnn_5/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
Г
model_2/simple_rnn_5/while/NoOpNoOpD^model_2/simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOpC^model_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOpE^model_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "S
#model_2_simple_rnn_5_while_identity,model_2/simple_rnn_5/while/Identity:output:0"W
%model_2_simple_rnn_5_while_identity_1.model_2/simple_rnn_5/while/Identity_1:output:0"W
%model_2_simple_rnn_5_while_identity_2.model_2/simple_rnn_5/while/Identity_2:output:0"W
%model_2_simple_rnn_5_while_identity_3.model_2/simple_rnn_5/while/Identity_3:output:0"W
%model_2_simple_rnn_5_while_identity_4.model_2/simple_rnn_5/while/Identity_4:output:0"
?model_2_simple_rnn_5_while_model_2_simple_rnn_5_strided_slice_1Amodel_2_simple_rnn_5_while_model_2_simple_rnn_5_strided_slice_1_0"
Lmodel_2_simple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resourceNmodel_2_simple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resource_0" 
Mmodel_2_simple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resourceOmodel_2_simple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0"
Kmodel_2_simple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resourceMmodel_2_simple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resource_0"ќ
{model_2_simple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_model_2_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor}model_2_simple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_model_2_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ
: : : : : 2
Cmodel_2/simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOpCmodel_2/simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp2
Bmodel_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOpBmodel_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp2
Dmodel_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpDmodel_2/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
: 
@
П
&model_2_simple_rnn_6_while_body_240217F
Bmodel_2_simple_rnn_6_while_model_2_simple_rnn_6_while_loop_counterL
Hmodel_2_simple_rnn_6_while_model_2_simple_rnn_6_while_maximum_iterations*
&model_2_simple_rnn_6_while_placeholder,
(model_2_simple_rnn_6_while_placeholder_1,
(model_2_simple_rnn_6_while_placeholder_2E
Amodel_2_simple_rnn_6_while_model_2_simple_rnn_6_strided_slice_1_0
}model_2_simple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_model_2_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor_0_
Mmodel_2_simple_rnn_6_while_simple_rnn_cell_6_matmul_readvariableop_resource_0:
\
Nmodel_2_simple_rnn_6_while_simple_rnn_cell_6_biasadd_readvariableop_resource_0:a
Omodel_2_simple_rnn_6_while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0:'
#model_2_simple_rnn_6_while_identity)
%model_2_simple_rnn_6_while_identity_1)
%model_2_simple_rnn_6_while_identity_2)
%model_2_simple_rnn_6_while_identity_3)
%model_2_simple_rnn_6_while_identity_4C
?model_2_simple_rnn_6_while_model_2_simple_rnn_6_strided_slice_1
{model_2_simple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_model_2_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor]
Kmodel_2_simple_rnn_6_while_simple_rnn_cell_6_matmul_readvariableop_resource:
Z
Lmodel_2_simple_rnn_6_while_simple_rnn_cell_6_biasadd_readvariableop_resource:_
Mmodel_2_simple_rnn_6_while_simple_rnn_cell_6_matmul_1_readvariableop_resource:ЂCmodel_2/simple_rnn_6/while/simple_rnn_cell_6/BiasAdd/ReadVariableOpЂBmodel_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul/ReadVariableOpЂDmodel_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp
Lmodel_2/simple_rnn_6/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   
>model_2/simple_rnn_6/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}model_2_simple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_model_2_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor_0&model_2_simple_rnn_6_while_placeholderUmodel_2/simple_rnn_6/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ
*
element_dtype0а
Bmodel_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul/ReadVariableOpReadVariableOpMmodel_2_simple_rnn_6_while_simple_rnn_cell_6_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype0
3model_2/simple_rnn_6/while/simple_rnn_cell_6/MatMulMatMulEmodel_2/simple_rnn_6/while/TensorArrayV2Read/TensorListGetItem:item:0Jmodel_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЮ
Cmodel_2/simple_rnn_6/while/simple_rnn_cell_6/BiasAdd/ReadVariableOpReadVariableOpNmodel_2_simple_rnn_6_while_simple_rnn_cell_6_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0§
4model_2/simple_rnn_6/while/simple_rnn_cell_6/BiasAddBiasAdd=model_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul:product:0Kmodel_2/simple_rnn_6/while/simple_rnn_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџд
Dmodel_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul_1/ReadVariableOpReadVariableOpOmodel_2_simple_rnn_6_while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0щ
5model_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul_1MatMul(model_2_simple_rnn_6_while_placeholder_2Lmodel_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџы
0model_2/simple_rnn_6/while/simple_rnn_cell_6/addAddV2=model_2/simple_rnn_6/while/simple_rnn_cell_6/BiasAdd:output:0?model_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџЁ
1model_2/simple_rnn_6/while/simple_rnn_cell_6/ReluRelu4model_2/simple_rnn_6/while/simple_rnn_cell_6/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџЇ
?model_2/simple_rnn_6/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(model_2_simple_rnn_6_while_placeholder_1&model_2_simple_rnn_6_while_placeholder?model_2/simple_rnn_6/while/simple_rnn_cell_6/Relu:activations:0*
_output_shapes
: *
element_dtype0:щшвb
 model_2/simple_rnn_6/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
model_2/simple_rnn_6/while/addAddV2&model_2_simple_rnn_6_while_placeholder)model_2/simple_rnn_6/while/add/y:output:0*
T0*
_output_shapes
: d
"model_2/simple_rnn_6/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Л
 model_2/simple_rnn_6/while/add_1AddV2Bmodel_2_simple_rnn_6_while_model_2_simple_rnn_6_while_loop_counter+model_2/simple_rnn_6/while/add_1/y:output:0*
T0*
_output_shapes
: 
#model_2/simple_rnn_6/while/IdentityIdentity$model_2/simple_rnn_6/while/add_1:z:0 ^model_2/simple_rnn_6/while/NoOp*
T0*
_output_shapes
: О
%model_2/simple_rnn_6/while/Identity_1IdentityHmodel_2_simple_rnn_6_while_model_2_simple_rnn_6_while_maximum_iterations ^model_2/simple_rnn_6/while/NoOp*
T0*
_output_shapes
: 
%model_2/simple_rnn_6/while/Identity_2Identity"model_2/simple_rnn_6/while/add:z:0 ^model_2/simple_rnn_6/while/NoOp*
T0*
_output_shapes
: Х
%model_2/simple_rnn_6/while/Identity_3IdentityOmodel_2/simple_rnn_6/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^model_2/simple_rnn_6/while/NoOp*
T0*
_output_shapes
: Ц
%model_2/simple_rnn_6/while/Identity_4Identity?model_2/simple_rnn_6/while/simple_rnn_cell_6/Relu:activations:0 ^model_2/simple_rnn_6/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџГ
model_2/simple_rnn_6/while/NoOpNoOpD^model_2/simple_rnn_6/while/simple_rnn_cell_6/BiasAdd/ReadVariableOpC^model_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul/ReadVariableOpE^model_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "S
#model_2_simple_rnn_6_while_identity,model_2/simple_rnn_6/while/Identity:output:0"W
%model_2_simple_rnn_6_while_identity_1.model_2/simple_rnn_6/while/Identity_1:output:0"W
%model_2_simple_rnn_6_while_identity_2.model_2/simple_rnn_6/while/Identity_2:output:0"W
%model_2_simple_rnn_6_while_identity_3.model_2/simple_rnn_6/while/Identity_3:output:0"W
%model_2_simple_rnn_6_while_identity_4.model_2/simple_rnn_6/while/Identity_4:output:0"
?model_2_simple_rnn_6_while_model_2_simple_rnn_6_strided_slice_1Amodel_2_simple_rnn_6_while_model_2_simple_rnn_6_strided_slice_1_0"
Lmodel_2_simple_rnn_6_while_simple_rnn_cell_6_biasadd_readvariableop_resourceNmodel_2_simple_rnn_6_while_simple_rnn_cell_6_biasadd_readvariableop_resource_0" 
Mmodel_2_simple_rnn_6_while_simple_rnn_cell_6_matmul_1_readvariableop_resourceOmodel_2_simple_rnn_6_while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0"
Kmodel_2_simple_rnn_6_while_simple_rnn_cell_6_matmul_readvariableop_resourceMmodel_2_simple_rnn_6_while_simple_rnn_cell_6_matmul_readvariableop_resource_0"ќ
{model_2_simple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_model_2_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor}model_2_simple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_model_2_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ: : : : : 2
Cmodel_2/simple_rnn_6/while/simple_rnn_cell_6/BiasAdd/ReadVariableOpCmodel_2/simple_rnn_6/while/simple_rnn_cell_6/BiasAdd/ReadVariableOp2
Bmodel_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul/ReadVariableOpBmodel_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul/ReadVariableOp2
Dmodel_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul_1/ReadVariableOpDmodel_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
я

C__inference_model_2_layer_call_and_return_conditional_losses_239709

inputs%
simple_rnn_5_239693:
!
simple_rnn_5_239695:
%
simple_rnn_5_239697:

%
simple_rnn_6_239701:
!
simple_rnn_6_239703:%
simple_rnn_6_239705:
identityЂ$simple_rnn_5/StatefulPartitionedCallЂ$simple_rnn_6/StatefulPartitionedCall
$simple_rnn_5/StatefulPartitionedCallStatefulPartitionedCallinputssimple_rnn_5_239693simple_rnn_5_239695simple_rnn_5_239697*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_239662ё
repeat_vector_2/PartitionedCallPartitionedCall-simple_rnn_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_repeat_vector_2_layer_call_and_return_conditional_losses_238858Н
$simple_rnn_6/StatefulPartitionedCallStatefulPartitionedCall(repeat_vector_2/PartitionedCall:output:0simple_rnn_6_239701simple_rnn_6_239703simple_rnn_6_239705*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_simple_rnn_6_layer_call_and_return_conditional_losses_239530
IdentityIdentity-simple_rnn_6/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ
NoOpNoOp%^simple_rnn_5/StatefulPartitionedCall%^simple_rnn_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : : : : : 2L
$simple_rnn_5/StatefulPartitionedCall$simple_rnn_5/StatefulPartitionedCall2L
$simple_rnn_6/StatefulPartitionedCall$simple_rnn_6/StatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
к
Њ
while_cond_241860
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_241860___redundant_placeholder04
0while_while_cond_241860___redundant_placeholder14
0while_while_cond_241860___redundant_placeholder24
0while_while_cond_241860___redundant_placeholder3
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
-: : : : :џџџџџџџџџ: ::::: 
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
Л
g
K__inference_repeat_vector_2_layer_call_and_return_conditional_losses_241559

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
valueB"         p
TileTileExpandDims:output:0stack:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџb
IdentityIdentityTile:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџџџџџџџџџџ:X T
0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
К

Ё
simple_rnn_5_while_cond_2406296
2simple_rnn_5_while_simple_rnn_5_while_loop_counter<
8simple_rnn_5_while_simple_rnn_5_while_maximum_iterations"
simple_rnn_5_while_placeholder$
 simple_rnn_5_while_placeholder_1$
 simple_rnn_5_while_placeholder_28
4simple_rnn_5_while_less_simple_rnn_5_strided_slice_1N
Jsimple_rnn_5_while_simple_rnn_5_while_cond_240629___redundant_placeholder0N
Jsimple_rnn_5_while_simple_rnn_5_while_cond_240629___redundant_placeholder1N
Jsimple_rnn_5_while_simple_rnn_5_while_cond_240629___redundant_placeholder2N
Jsimple_rnn_5_while_simple_rnn_5_while_cond_240629___redundant_placeholder3
simple_rnn_5_while_identity

simple_rnn_5/while/LessLesssimple_rnn_5_while_placeholder4simple_rnn_5_while_less_simple_rnn_5_strided_slice_1*
T0*
_output_shapes
: e
simple_rnn_5/while/IdentityIdentitysimple_rnn_5/while/Less:z:0*
T0
*
_output_shapes
: "C
simple_rnn_5_while_identity$simple_rnn_5/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :џџџџџџџџџ
: ::::: 
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
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
:

Р
C__inference_model_2_layer_call_and_return_conditional_losses_241023

inputsO
=simple_rnn_5_simple_rnn_cell_5_matmul_readvariableop_resource:
L
>simple_rnn_5_simple_rnn_cell_5_biasadd_readvariableop_resource:
Q
?simple_rnn_5_simple_rnn_cell_5_matmul_1_readvariableop_resource:

O
=simple_rnn_6_simple_rnn_cell_6_matmul_readvariableop_resource:
L
>simple_rnn_6_simple_rnn_cell_6_biasadd_readvariableop_resource:Q
?simple_rnn_6_simple_rnn_cell_6_matmul_1_readvariableop_resource:
identityЂ5simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOpЂ4simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOpЂ6simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOpЂsimple_rnn_5/whileЂ5simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOpЂ4simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOpЂ6simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOpЂsimple_rnn_6/whileH
simple_rnn_5/ShapeShapeinputs*
T0*
_output_shapes
:j
 simple_rnn_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"simple_rnn_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"simple_rnn_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
simple_rnn_5/strided_sliceStridedSlicesimple_rnn_5/Shape:output:0)simple_rnn_5/strided_slice/stack:output:0+simple_rnn_5/strided_slice/stack_1:output:0+simple_rnn_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
simple_rnn_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :

simple_rnn_5/zeros/packedPack#simple_rnn_5/strided_slice:output:0$simple_rnn_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
simple_rnn_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
simple_rnn_5/zerosFill"simple_rnn_5/zeros/packed:output:0!simple_rnn_5/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
p
simple_rnn_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
simple_rnn_5/transpose	Transposeinputs$simple_rnn_5/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ^
simple_rnn_5/Shape_1Shapesimple_rnn_5/transpose:y:0*
T0*
_output_shapes
:l
"simple_rnn_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
simple_rnn_5/strided_slice_1StridedSlicesimple_rnn_5/Shape_1:output:0+simple_rnn_5/strided_slice_1/stack:output:0-simple_rnn_5/strided_slice_1/stack_1:output:0-simple_rnn_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(simple_rnn_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџл
simple_rnn_5/TensorArrayV2TensorListReserve1simple_rnn_5/TensorArrayV2/element_shape:output:0%simple_rnn_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
Bsimple_rnn_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
4simple_rnn_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_5/transpose:y:0Ksimple_rnn_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвl
"simple_rnn_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Њ
simple_rnn_5/strided_slice_2StridedSlicesimple_rnn_5/transpose:y:0+simple_rnn_5/strided_slice_2/stack:output:0-simple_rnn_5/strided_slice_2/stack_1:output:0-simple_rnn_5/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskВ
4simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp=simple_rnn_5_simple_rnn_cell_5_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0Ц
%simple_rnn_5/simple_rnn_cell_5/MatMulMatMul%simple_rnn_5/strided_slice_2:output:0<simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
А
5simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_5_simple_rnn_cell_5_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0г
&simple_rnn_5/simple_rnn_cell_5/BiasAddBiasAdd/simple_rnn_5/simple_rnn_cell_5/MatMul:product:0=simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Ж
6simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_5_simple_rnn_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype0Р
'simple_rnn_5/simple_rnn_cell_5/MatMul_1MatMulsimple_rnn_5/zeros:output:0>simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
С
"simple_rnn_5/simple_rnn_cell_5/addAddV2/simple_rnn_5/simple_rnn_cell_5/BiasAdd:output:01simple_rnn_5/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ

#simple_rnn_5/simple_rnn_cell_5/ReluRelu&simple_rnn_5/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
{
*simple_rnn_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   k
)simple_rnn_5/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :ь
simple_rnn_5/TensorArrayV2_1TensorListReserve3simple_rnn_5/TensorArrayV2_1/element_shape:output:02simple_rnn_5/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвS
simple_rnn_5/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%simple_rnn_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџa
simple_rnn_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
simple_rnn_5/whileWhile(simple_rnn_5/while/loop_counter:output:0.simple_rnn_5/while/maximum_iterations:output:0simple_rnn_5/time:output:0%simple_rnn_5/TensorArrayV2_1:handle:0simple_rnn_5/zeros:output:0%simple_rnn_5/strided_slice_1:output:0Dsimple_rnn_5/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_5_simple_rnn_cell_5_matmul_readvariableop_resource>simple_rnn_5_simple_rnn_cell_5_biasadd_readvariableop_resource?simple_rnn_5_simple_rnn_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( **
body"R 
simple_rnn_5_while_body_240848**
cond"R 
simple_rnn_5_while_cond_240847*8
output_shapes'
%: : : : :џџџџџџџџџ
: : : : : *
parallel_iterations 
=simple_rnn_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   §
/simple_rnn_5/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_5/while:output:3Fsimple_rnn_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ
*
element_dtype0*
num_elementsu
"simple_rnn_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџn
$simple_rnn_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ш
simple_rnn_5/strided_slice_3StridedSlice8simple_rnn_5/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_5/strided_slice_3/stack:output:0-simple_rnn_5/strided_slice_3/stack_1:output:0-simple_rnn_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ
*
shrink_axis_maskr
simple_rnn_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Н
simple_rnn_5/transpose_1	Transpose8simple_rnn_5/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_5/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
`
repeat_vector_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Ў
repeat_vector_2/ExpandDims
ExpandDims%simple_rnn_5/strided_slice_3:output:0'repeat_vector_2/ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
j
repeat_vector_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"         
repeat_vector_2/TileTile#repeat_vector_2/ExpandDims:output:0repeat_vector_2/stack:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
_
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
value	B :
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
:џџџџџџџџџp
simple_rnn_6/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
simple_rnn_6/transpose	Transposerepeat_vector_2/Tile:output:0$simple_rnn_6/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
^
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
valueB"џџџџ
   
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
valueB:Њ
simple_rnn_6/strided_slice_2StridedSlicesimple_rnn_6/transpose:y:0+simple_rnn_6/strided_slice_2/stack:output:0-simple_rnn_6/strided_slice_2/stack_1:output:0-simple_rnn_6/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ
*
shrink_axis_maskВ
4simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOpReadVariableOp=simple_rnn_6_simple_rnn_cell_6_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0Ц
%simple_rnn_6/simple_rnn_cell_6/MatMulMatMul%simple_rnn_6/strided_slice_2:output:0<simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџА
5simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_6_simple_rnn_cell_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0г
&simple_rnn_6/simple_rnn_cell_6/BiasAddBiasAdd/simple_rnn_6/simple_rnn_cell_6/MatMul:product:0=simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЖ
6simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_6_simple_rnn_cell_6_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0Р
'simple_rnn_6/simple_rnn_cell_6/MatMul_1MatMulsimple_rnn_6/zeros:output:0>simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџС
"simple_rnn_6/simple_rnn_cell_6/addAddV2/simple_rnn_6/simple_rnn_cell_6/BiasAdd:output:01simple_rnn_6/simple_rnn_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ
#simple_rnn_6/simple_rnn_cell_6/ReluRelu&simple_rnn_6/simple_rnn_cell_6/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ{
*simple_rnn_6/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   п
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
value	B : 
simple_rnn_6/whileWhile(simple_rnn_6/while/loop_counter:output:0.simple_rnn_6/while/maximum_iterations:output:0simple_rnn_6/time:output:0%simple_rnn_6/TensorArrayV2_1:handle:0simple_rnn_6/zeros:output:0%simple_rnn_6/strided_slice_1:output:0Dsimple_rnn_6/TensorArrayUnstack/TensorListFromTensor:output_handle:0=simple_rnn_6_simple_rnn_cell_6_matmul_readvariableop_resource>simple_rnn_6_simple_rnn_cell_6_biasadd_readvariableop_resource?simple_rnn_6_simple_rnn_cell_6_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( **
body"R 
simple_rnn_6_while_body_240957**
cond"R 
simple_rnn_6_while_cond_240956*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 
=simple_rnn_6/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   щ
/simple_rnn_6/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_6/while:output:3Fsimple_rnn_6/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
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
:џџџџџџџџџ*
shrink_axis_maskr
simple_rnn_6/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Н
simple_rnn_6/transpose_1	Transpose8simple_rnn_6/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_6/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџo
IdentityIdentitysimple_rnn_6/transpose_1:y:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџР
NoOpNoOp6^simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp5^simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp7^simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp^simple_rnn_5/while6^simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOp5^simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOp7^simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOp^simple_rnn_6/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : : : : : 2n
5simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp5simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp2l
4simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp4simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp2p
6simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp6simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp2(
simple_rnn_5/whilesimple_rnn_5/while2n
5simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOp5simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOp2l
4simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOp4simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOp2p
6simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOp6simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOp2(
simple_rnn_6/whilesimple_rnn_6/while:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ў
З
-__inference_simple_rnn_5_layer_call_fn_241106

inputs
unknown:

	unknown_0:

	unknown_1:


identityЂStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_239662o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
4

H__inference_simple_rnn_6_layer_call_and_return_conditional_losses_239144

inputs*
simple_rnn_cell_6_239069:
&
simple_rnn_cell_6_239071:*
simple_rnn_cell_6_239073:
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
value	B :s
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
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
D
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
valueB"џџџџ
   р
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ
*
shrink_axis_maskч
)simple_rnn_cell_6/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_6_239069simple_rnn_cell_6_239071simple_rnn_cell_6_239073*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_simple_rnn_cell_6_layer_call_and_return_conditional_losses_239029n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
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
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_6_239069simple_rnn_cell_6_239071simple_rnn_cell_6_239073*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_239081*
condR
while_cond_239080*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ы
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
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
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџk
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџz
NoOpNoOp*^simple_rnn_cell_6/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ
: : : 2V
)simple_rnn_cell_6/StatefulPartitionedCall)simple_rnn_cell_6/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ

 
_user_specified_nameinputs
ђ

C__inference_model_2_layer_call_and_return_conditional_losses_239779
input_6%
simple_rnn_5_239763:
!
simple_rnn_5_239765:
%
simple_rnn_5_239767:

%
simple_rnn_6_239771:
!
simple_rnn_6_239773:%
simple_rnn_6_239775:
identityЂ$simple_rnn_5/StatefulPartitionedCallЂ$simple_rnn_6/StatefulPartitionedCall
$simple_rnn_5/StatefulPartitionedCallStatefulPartitionedCallinput_6simple_rnn_5_239763simple_rnn_5_239765simple_rnn_5_239767*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_239662ё
repeat_vector_2/PartitionedCallPartitionedCall-simple_rnn_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_repeat_vector_2_layer_call_and_return_conditional_losses_238858Н
$simple_rnn_6/StatefulPartitionedCallStatefulPartitionedCall(repeat_vector_2/PartitionedCall:output:0simple_rnn_6_239771simple_rnn_6_239773simple_rnn_6_239775*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_simple_rnn_6_layer_call_and_return_conditional_losses_239530
IdentityIdentity-simple_rnn_6/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ
NoOpNoOp%^simple_rnn_5/StatefulPartitionedCall%^simple_rnn_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : : : : : 2L
$simple_rnn_5/StatefulPartitionedCall$simple_rnn_5/StatefulPartitionedCall2L
$simple_rnn_6/StatefulPartitionedCall$simple_rnn_6/StatefulPartitionedCall:T P
+
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_6

ш
M__inference_simple_rnn_cell_6_layer_call_and_return_conditional_losses_239029

inputs

states0
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:2
 matmul_1_readvariableop_resource:
identity

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
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
:џџџџџџџџџx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџG
ReluReluadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџc

Identity_1IdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ
:џџџџџџџџџ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_namestates

ъ
M__inference_simple_rnn_cell_6_layer_call_and_return_conditional_losses_242142

inputs
states_00
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:2
 matmul_1_readvariableop_resource:
identity

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
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
:џџџџџџџџџx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџG
ReluReluadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџc

Identity_1IdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ
:џџџџџџџџџ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0
ј4

H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_238837

inputs*
simple_rnn_cell_5_238760:
&
simple_rnn_cell_5_238762:
*
simple_rnn_cell_5_238764:


identityЂ)simple_rnn_cell_5/StatefulPartitionedCallЂwhile;
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
value	B :
s
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
:џџџџџџџџџ
c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџD
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
valueB"џџџџ   р
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskч
)simple_rnn_cell_5/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_5_238760simple_rnn_cell_5_238762simple_rnn_cell_5_238764*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ
:џџџџџџџџџ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_238720n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   ^
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
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_5_238760simple_rnn_cell_5_238762simple_rnn_cell_5_238764*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_238773*
condR
while_cond_238772*8
output_shapes'
%: : : : :џџџџџџџџџ
: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ
*
element_dtype0*
num_elementsh
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
:џџџџџџџџџ
*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
z
NoOpNoOp*^simple_rnn_cell_5/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџџџџџџџџџџ: : : 2V
)simple_rnn_cell_5/StatefulPartitionedCall)simple_rnn_cell_5/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ќ

(__inference_model_2_layer_call_fn_240570

inputs
unknown:

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:
	unknown_4:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_239394s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ў
З
-__inference_simple_rnn_5_layer_call_fn_241095

inputs
unknown:

	unknown_0:

	unknown_1:


identityЂStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_239269o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ЈТ
э	
H__inference_sequential_5_layer_call_and_return_conditional_losses_240309

inputsW
Emodel_2_simple_rnn_5_simple_rnn_cell_5_matmul_readvariableop_resource:
T
Fmodel_2_simple_rnn_5_simple_rnn_cell_5_biasadd_readvariableop_resource:
Y
Gmodel_2_simple_rnn_5_simple_rnn_cell_5_matmul_1_readvariableop_resource:

W
Emodel_2_simple_rnn_6_simple_rnn_cell_6_matmul_readvariableop_resource:
T
Fmodel_2_simple_rnn_6_simple_rnn_cell_6_biasadd_readvariableop_resource:Y
Gmodel_2_simple_rnn_6_simple_rnn_cell_6_matmul_1_readvariableop_resource:;
)dense_2_tensordot_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:
identityЂdense_2/BiasAdd/ReadVariableOpЂ dense_2/Tensordot/ReadVariableOpЂ=model_2/simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOpЂ<model_2/simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOpЂ>model_2/simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOpЂmodel_2/simple_rnn_5/whileЂ=model_2/simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOpЂ<model_2/simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOpЂ>model_2/simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOpЂmodel_2/simple_rnn_6/whileP
model_2/simple_rnn_5/ShapeShapeinputs*
T0*
_output_shapes
:r
(model_2/simple_rnn_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*model_2/simple_rnn_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*model_2/simple_rnn_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:К
"model_2/simple_rnn_5/strided_sliceStridedSlice#model_2/simple_rnn_5/Shape:output:01model_2/simple_rnn_5/strided_slice/stack:output:03model_2/simple_rnn_5/strided_slice/stack_1:output:03model_2/simple_rnn_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#model_2/simple_rnn_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
В
!model_2/simple_rnn_5/zeros/packedPack+model_2/simple_rnn_5/strided_slice:output:0,model_2/simple_rnn_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:e
 model_2/simple_rnn_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ћ
model_2/simple_rnn_5/zerosFill*model_2/simple_rnn_5/zeros/packed:output:0)model_2/simple_rnn_5/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
x
#model_2/simple_rnn_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
model_2/simple_rnn_5/transpose	Transposeinputs,model_2/simple_rnn_5/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџn
model_2/simple_rnn_5/Shape_1Shape"model_2/simple_rnn_5/transpose:y:0*
T0*
_output_shapes
:t
*model_2/simple_rnn_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,model_2/simple_rnn_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,model_2/simple_rnn_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ф
$model_2/simple_rnn_5/strided_slice_1StridedSlice%model_2/simple_rnn_5/Shape_1:output:03model_2/simple_rnn_5/strided_slice_1/stack:output:05model_2/simple_rnn_5/strided_slice_1/stack_1:output:05model_2/simple_rnn_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
0model_2/simple_rnn_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџѓ
"model_2/simple_rnn_5/TensorArrayV2TensorListReserve9model_2/simple_rnn_5/TensorArrayV2/element_shape:output:0-model_2/simple_rnn_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
Jmodel_2/simple_rnn_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
<model_2/simple_rnn_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"model_2/simple_rnn_5/transpose:y:0Smodel_2/simple_rnn_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвt
*model_2/simple_rnn_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,model_2/simple_rnn_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,model_2/simple_rnn_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:в
$model_2/simple_rnn_5/strided_slice_2StridedSlice"model_2/simple_rnn_5/transpose:y:03model_2/simple_rnn_5/strided_slice_2/stack:output:05model_2/simple_rnn_5/strided_slice_2/stack_1:output:05model_2/simple_rnn_5/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskТ
<model_2/simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOpEmodel_2_simple_rnn_5_simple_rnn_cell_5_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0о
-model_2/simple_rnn_5/simple_rnn_cell_5/MatMulMatMul-model_2/simple_rnn_5/strided_slice_2:output:0Dmodel_2/simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Р
=model_2/simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOpFmodel_2_simple_rnn_5_simple_rnn_cell_5_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0ы
.model_2/simple_rnn_5/simple_rnn_cell_5/BiasAddBiasAdd7model_2/simple_rnn_5/simple_rnn_cell_5/MatMul:product:0Emodel_2/simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Ц
>model_2/simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOpGmodel_2_simple_rnn_5_simple_rnn_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype0и
/model_2/simple_rnn_5/simple_rnn_cell_5/MatMul_1MatMul#model_2/simple_rnn_5/zeros:output:0Fmodel_2/simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
й
*model_2/simple_rnn_5/simple_rnn_cell_5/addAddV27model_2/simple_rnn_5/simple_rnn_cell_5/BiasAdd:output:09model_2/simple_rnn_5/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ

+model_2/simple_rnn_5/simple_rnn_cell_5/ReluRelu.model_2/simple_rnn_5/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ

2model_2/simple_rnn_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   s
1model_2/simple_rnn_5/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
$model_2/simple_rnn_5/TensorArrayV2_1TensorListReserve;model_2/simple_rnn_5/TensorArrayV2_1/element_shape:output:0:model_2/simple_rnn_5/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв[
model_2/simple_rnn_5/timeConst*
_output_shapes
: *
dtype0*
value	B : x
-model_2/simple_rnn_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџi
'model_2/simple_rnn_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : щ
model_2/simple_rnn_5/whileWhile0model_2/simple_rnn_5/while/loop_counter:output:06model_2/simple_rnn_5/while/maximum_iterations:output:0"model_2/simple_rnn_5/time:output:0-model_2/simple_rnn_5/TensorArrayV2_1:handle:0#model_2/simple_rnn_5/zeros:output:0-model_2/simple_rnn_5/strided_slice_1:output:0Lmodel_2/simple_rnn_5/TensorArrayUnstack/TensorListFromTensor:output_handle:0Emodel_2_simple_rnn_5_simple_rnn_cell_5_matmul_readvariableop_resourceFmodel_2_simple_rnn_5_simple_rnn_cell_5_biasadd_readvariableop_resourceGmodel_2_simple_rnn_5_simple_rnn_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *2
body*R(
&model_2_simple_rnn_5_while_body_240108*2
cond*R(
&model_2_simple_rnn_5_while_cond_240107*8
output_shapes'
%: : : : :џџџџџџџџџ
: : : : : *
parallel_iterations 
Emodel_2/simple_rnn_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   
7model_2/simple_rnn_5/TensorArrayV2Stack/TensorListStackTensorListStack#model_2/simple_rnn_5/while:output:3Nmodel_2/simple_rnn_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ
*
element_dtype0*
num_elements}
*model_2/simple_rnn_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџv
,model_2/simple_rnn_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,model_2/simple_rnn_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:№
$model_2/simple_rnn_5/strided_slice_3StridedSlice@model_2/simple_rnn_5/TensorArrayV2Stack/TensorListStack:tensor:03model_2/simple_rnn_5/strided_slice_3/stack:output:05model_2/simple_rnn_5/strided_slice_3/stack_1:output:05model_2/simple_rnn_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ
*
shrink_axis_maskz
%model_2/simple_rnn_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          е
 model_2/simple_rnn_5/transpose_1	Transpose@model_2/simple_rnn_5/TensorArrayV2Stack/TensorListStack:tensor:0.model_2/simple_rnn_5/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
h
&model_2/repeat_vector_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
"model_2/repeat_vector_2/ExpandDims
ExpandDims-model_2/simple_rnn_5/strided_slice_3:output:0/model_2/repeat_vector_2/ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
r
model_2/repeat_vector_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"         Џ
model_2/repeat_vector_2/TileTile+model_2/repeat_vector_2/ExpandDims:output:0&model_2/repeat_vector_2/stack:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
o
model_2/simple_rnn_6/ShapeShape%model_2/repeat_vector_2/Tile:output:0*
T0*
_output_shapes
:r
(model_2/simple_rnn_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*model_2/simple_rnn_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*model_2/simple_rnn_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:К
"model_2/simple_rnn_6/strided_sliceStridedSlice#model_2/simple_rnn_6/Shape:output:01model_2/simple_rnn_6/strided_slice/stack:output:03model_2/simple_rnn_6/strided_slice/stack_1:output:03model_2/simple_rnn_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#model_2/simple_rnn_6/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :В
!model_2/simple_rnn_6/zeros/packedPack+model_2/simple_rnn_6/strided_slice:output:0,model_2/simple_rnn_6/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:e
 model_2/simple_rnn_6/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ћ
model_2/simple_rnn_6/zerosFill*model_2/simple_rnn_6/zeros/packed:output:0)model_2/simple_rnn_6/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџx
#model_2/simple_rnn_6/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ж
model_2/simple_rnn_6/transpose	Transpose%model_2/repeat_vector_2/Tile:output:0,model_2/simple_rnn_6/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
n
model_2/simple_rnn_6/Shape_1Shape"model_2/simple_rnn_6/transpose:y:0*
T0*
_output_shapes
:t
*model_2/simple_rnn_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,model_2/simple_rnn_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,model_2/simple_rnn_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ф
$model_2/simple_rnn_6/strided_slice_1StridedSlice%model_2/simple_rnn_6/Shape_1:output:03model_2/simple_rnn_6/strided_slice_1/stack:output:05model_2/simple_rnn_6/strided_slice_1/stack_1:output:05model_2/simple_rnn_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
0model_2/simple_rnn_6/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџѓ
"model_2/simple_rnn_6/TensorArrayV2TensorListReserve9model_2/simple_rnn_6/TensorArrayV2/element_shape:output:0-model_2/simple_rnn_6/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
Jmodel_2/simple_rnn_6/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   
<model_2/simple_rnn_6/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"model_2/simple_rnn_6/transpose:y:0Smodel_2/simple_rnn_6/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвt
*model_2/simple_rnn_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,model_2/simple_rnn_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,model_2/simple_rnn_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:в
$model_2/simple_rnn_6/strided_slice_2StridedSlice"model_2/simple_rnn_6/transpose:y:03model_2/simple_rnn_6/strided_slice_2/stack:output:05model_2/simple_rnn_6/strided_slice_2/stack_1:output:05model_2/simple_rnn_6/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ
*
shrink_axis_maskТ
<model_2/simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOpReadVariableOpEmodel_2_simple_rnn_6_simple_rnn_cell_6_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0о
-model_2/simple_rnn_6/simple_rnn_cell_6/MatMulMatMul-model_2/simple_rnn_6/strided_slice_2:output:0Dmodel_2/simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџР
=model_2/simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOpReadVariableOpFmodel_2_simple_rnn_6_simple_rnn_cell_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ы
.model_2/simple_rnn_6/simple_rnn_cell_6/BiasAddBiasAdd7model_2/simple_rnn_6/simple_rnn_cell_6/MatMul:product:0Emodel_2/simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЦ
>model_2/simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOpReadVariableOpGmodel_2_simple_rnn_6_simple_rnn_cell_6_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0и
/model_2/simple_rnn_6/simple_rnn_cell_6/MatMul_1MatMul#model_2/simple_rnn_6/zeros:output:0Fmodel_2/simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџй
*model_2/simple_rnn_6/simple_rnn_cell_6/addAddV27model_2/simple_rnn_6/simple_rnn_cell_6/BiasAdd:output:09model_2/simple_rnn_6/simple_rnn_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ
+model_2/simple_rnn_6/simple_rnn_cell_6/ReluRelu.model_2/simple_rnn_6/simple_rnn_cell_6/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
2model_2/simple_rnn_6/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ї
$model_2/simple_rnn_6/TensorArrayV2_1TensorListReserve;model_2/simple_rnn_6/TensorArrayV2_1/element_shape:output:0-model_2/simple_rnn_6/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв[
model_2/simple_rnn_6/timeConst*
_output_shapes
: *
dtype0*
value	B : x
-model_2/simple_rnn_6/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџi
'model_2/simple_rnn_6/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : щ
model_2/simple_rnn_6/whileWhile0model_2/simple_rnn_6/while/loop_counter:output:06model_2/simple_rnn_6/while/maximum_iterations:output:0"model_2/simple_rnn_6/time:output:0-model_2/simple_rnn_6/TensorArrayV2_1:handle:0#model_2/simple_rnn_6/zeros:output:0-model_2/simple_rnn_6/strided_slice_1:output:0Lmodel_2/simple_rnn_6/TensorArrayUnstack/TensorListFromTensor:output_handle:0Emodel_2_simple_rnn_6_simple_rnn_cell_6_matmul_readvariableop_resourceFmodel_2_simple_rnn_6_simple_rnn_cell_6_biasadd_readvariableop_resourceGmodel_2_simple_rnn_6_simple_rnn_cell_6_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *2
body*R(
&model_2_simple_rnn_6_while_body_240217*2
cond*R(
&model_2_simple_rnn_6_while_cond_240216*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 
Emodel_2/simple_rnn_6/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
7model_2/simple_rnn_6/TensorArrayV2Stack/TensorListStackTensorListStack#model_2/simple_rnn_6/while:output:3Nmodel_2/simple_rnn_6/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype0}
*model_2/simple_rnn_6/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџv
,model_2/simple_rnn_6/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,model_2/simple_rnn_6/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:№
$model_2/simple_rnn_6/strided_slice_3StridedSlice@model_2/simple_rnn_6/TensorArrayV2Stack/TensorListStack:tensor:03model_2/simple_rnn_6/strided_slice_3/stack:output:05model_2/simple_rnn_6/strided_slice_3/stack_1:output:05model_2/simple_rnn_6/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskz
%model_2/simple_rnn_6/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          е
 model_2/simple_rnn_6/transpose_1	Transpose@model_2/simple_rnn_6/TensorArrayV2Stack/TensorListStack:tensor:0.model_2/simple_rnn_6/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       k
dense_2/Tensordot/ShapeShape$model_2/simple_rnn_6/transpose_1:y:0*
T0*
_output_shapes
:a
dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : л
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : п
dense_2/Tensordot/GatherV2_1GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/axes:output:0*dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : М
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ї
dense_2/Tensordot/transpose	Transpose$model_2/simple_rnn_6/transpose_1:y:0!dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџЂ
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЂ
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџc
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ч
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџk
IdentityIdentitydense_2/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџФ
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp>^model_2/simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp=^model_2/simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp?^model_2/simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp^model_2/simple_rnn_5/while>^model_2/simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOp=^model_2/simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOp?^model_2/simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOp^model_2/simple_rnn_6/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : : : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2~
=model_2/simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp=model_2/simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp2|
<model_2/simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp<model_2/simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp2
>model_2/simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp>model_2/simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp28
model_2/simple_rnn_5/whilemodel_2/simple_rnn_5/while2~
=model_2/simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOp=model_2/simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOp2|
<model_2/simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOp<model_2/simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOp2
>model_2/simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOp>model_2/simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOp28
model_2/simple_rnn_6/whilemodel_2/simple_rnn_6/while:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

ъ
M__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_242097

inputs
states_00
matmul_readvariableop_resource:
-
biasadd_readvariableop_resource:
2
 matmul_1_readvariableop_resource:


identity

identity_1ЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ
G
ReluReluadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
c

Identity_1IdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ

NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ:џџџџџџџџџ
: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ

"
_user_specified_name
states/0
ю
Џ
H__inference_sequential_5_layer_call_and_return_conditional_losses_239836

inputs 
model_2_239786:

model_2_239788:
 
model_2_239790:

 
model_2_239792:

model_2_239794: 
model_2_239796: 
dense_2_239830:
dense_2_239832:
identityЂdense_2/StatefulPartitionedCallЂmodel_2/StatefulPartitionedCallИ
model_2/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_2_239786model_2_239788model_2_239790model_2_239792model_2_239794model_2_239796*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_239394
dense_2/StatefulPartitionedCallStatefulPartitionedCall(model_2/StatefulPartitionedCall:output:0dense_2_239830dense_2_239832*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_239829{
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ
NoOpNoOp ^dense_2/StatefulPartitionedCall ^model_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
model_2/StatefulPartitionedCallmodel_2/StatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
П	
О
$__inference_signature_wrapper_240023
model_2_input
unknown:

	unknown_0:

	unknown_1:


	unknown_2:

	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallmodel_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_238550s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:џџџџџџџџџ
'
_user_specified_namemodel_2_input
я

C__inference_model_2_layer_call_and_return_conditional_losses_239394

inputs%
simple_rnn_5_239270:
!
simple_rnn_5_239272:
%
simple_rnn_5_239274:

%
simple_rnn_6_239386:
!
simple_rnn_6_239388:%
simple_rnn_6_239390:
identityЂ$simple_rnn_5/StatefulPartitionedCallЂ$simple_rnn_6/StatefulPartitionedCall
$simple_rnn_5/StatefulPartitionedCallStatefulPartitionedCallinputssimple_rnn_5_239270simple_rnn_5_239272simple_rnn_5_239274*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_239269ё
repeat_vector_2/PartitionedCallPartitionedCall-simple_rnn_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_repeat_vector_2_layer_call_and_return_conditional_losses_238858Н
$simple_rnn_6/StatefulPartitionedCallStatefulPartitionedCall(repeat_vector_2/PartitionedCall:output:0simple_rnn_6_239386simple_rnn_6_239388simple_rnn_6_239390*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_simple_rnn_6_layer_call_and_return_conditional_losses_239385
IdentityIdentity-simple_rnn_6/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ
NoOpNoOp%^simple_rnn_5/StatefulPartitionedCall%^simple_rnn_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ: : : : : : 2L
$simple_rnn_5/StatefulPartitionedCall$simple_rnn_5/StatefulPartitionedCall2L
$simple_rnn_6/StatefulPartitionedCall$simple_rnn_6/StatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ќ9
Ю
simple_rnn_5_while_body_2406306
2simple_rnn_5_while_simple_rnn_5_while_loop_counter<
8simple_rnn_5_while_simple_rnn_5_while_maximum_iterations"
simple_rnn_5_while_placeholder$
 simple_rnn_5_while_placeholder_1$
 simple_rnn_5_while_placeholder_25
1simple_rnn_5_while_simple_rnn_5_strided_slice_1_0q
msimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0W
Esimple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resource_0:
T
Fsimple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resource_0:
Y
Gsimple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0:


simple_rnn_5_while_identity!
simple_rnn_5_while_identity_1!
simple_rnn_5_while_identity_2!
simple_rnn_5_while_identity_3!
simple_rnn_5_while_identity_43
/simple_rnn_5_while_simple_rnn_5_strided_slice_1o
ksimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensorU
Csimple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resource:
R
Dsimple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resource:
W
Esimple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource:

Ђ;simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOpЂ:simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOpЂ<simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp
Dsimple_rnn_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ч
6simple_rnn_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_5_while_placeholderMsimple_rnn_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ*
element_dtype0Р
:simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype0ъ
+simple_rnn_5/while/simple_rnn_cell_5/MatMulMatMul=simple_rnn_5/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
О
;simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resource_0*
_output_shapes
:
*
dtype0х
,simple_rnn_5/while/simple_rnn_cell_5/BiasAddBiasAdd5simple_rnn_5/while/simple_rnn_cell_5/MatMul:product:0Csimple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Ф
<simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:

*
dtype0б
-simple_rnn_5/while/simple_rnn_cell_5/MatMul_1MatMul simple_rnn_5_while_placeholder_2Dsimple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
г
(simple_rnn_5/while/simple_rnn_cell_5/addAddV25simple_rnn_5/while/simple_rnn_cell_5/BiasAdd:output:07simple_rnn_5/while/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ

)simple_rnn_5/while/simple_rnn_cell_5/ReluRelu,simple_rnn_5/while/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ

=simple_rnn_5/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : Џ
7simple_rnn_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_5_while_placeholder_1Fsimple_rnn_5/while/TensorArrayV2Write/TensorListSetItem/index:output:07simple_rnn_5/while/simple_rnn_cell_5/Relu:activations:0*
_output_shapes
: *
element_dtype0:щшвZ
simple_rnn_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
simple_rnn_5/while/addAddV2simple_rnn_5_while_placeholder!simple_rnn_5/while/add/y:output:0*
T0*
_output_shapes
: \
simple_rnn_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
simple_rnn_5/while/add_1AddV22simple_rnn_5_while_simple_rnn_5_while_loop_counter#simple_rnn_5/while/add_1/y:output:0*
T0*
_output_shapes
: 
simple_rnn_5/while/IdentityIdentitysimple_rnn_5/while/add_1:z:0^simple_rnn_5/while/NoOp*
T0*
_output_shapes
: 
simple_rnn_5/while/Identity_1Identity8simple_rnn_5_while_simple_rnn_5_while_maximum_iterations^simple_rnn_5/while/NoOp*
T0*
_output_shapes
: 
simple_rnn_5/while/Identity_2Identitysimple_rnn_5/while/add:z:0^simple_rnn_5/while/NoOp*
T0*
_output_shapes
: ­
simple_rnn_5/while/Identity_3IdentityGsimple_rnn_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_5/while/NoOp*
T0*
_output_shapes
: Ў
simple_rnn_5/while/Identity_4Identity7simple_rnn_5/while/simple_rnn_cell_5/Relu:activations:0^simple_rnn_5/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџ

simple_rnn_5/while/NoOpNoOp<^simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp;^simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp=^simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "C
simple_rnn_5_while_identity$simple_rnn_5/while/Identity:output:0"G
simple_rnn_5_while_identity_1&simple_rnn_5/while/Identity_1:output:0"G
simple_rnn_5_while_identity_2&simple_rnn_5/while/Identity_2:output:0"G
simple_rnn_5_while_identity_3&simple_rnn_5/while/Identity_3:output:0"G
simple_rnn_5_while_identity_4&simple_rnn_5/while/Identity_4:output:0"d
/simple_rnn_5_while_simple_rnn_5_strided_slice_11simple_rnn_5_while_simple_rnn_5_strided_slice_1_0"
Dsimple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resourceFsimple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resource_0"
Esimple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resourceGsimple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0"
Csimple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resourceEsimple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resource_0"м
ksimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensormsimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ
: : : : : 2z
;simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp;simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp2x
:simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp:simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp2|
<simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp<simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
: 
К

Ё
simple_rnn_6_while_cond_2409566
2simple_rnn_6_while_simple_rnn_6_while_loop_counter<
8simple_rnn_6_while_simple_rnn_6_while_maximum_iterations"
simple_rnn_6_while_placeholder$
 simple_rnn_6_while_placeholder_1$
 simple_rnn_6_while_placeholder_28
4simple_rnn_6_while_less_simple_rnn_6_strided_slice_1N
Jsimple_rnn_6_while_simple_rnn_6_while_cond_240956___redundant_placeholder0N
Jsimple_rnn_6_while_simple_rnn_6_while_cond_240956___redundant_placeholder1N
Jsimple_rnn_6_while_simple_rnn_6_while_cond_240956___redundant_placeholder2N
Jsimple_rnn_6_while_simple_rnn_6_while_cond_240956___redundant_placeholder3
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
-: : : : :џџџџџџџџџ: ::::: 
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
ЈТ
э	
H__inference_sequential_5_layer_call_and_return_conditional_losses_240553

inputsW
Emodel_2_simple_rnn_5_simple_rnn_cell_5_matmul_readvariableop_resource:
T
Fmodel_2_simple_rnn_5_simple_rnn_cell_5_biasadd_readvariableop_resource:
Y
Gmodel_2_simple_rnn_5_simple_rnn_cell_5_matmul_1_readvariableop_resource:

W
Emodel_2_simple_rnn_6_simple_rnn_cell_6_matmul_readvariableop_resource:
T
Fmodel_2_simple_rnn_6_simple_rnn_cell_6_biasadd_readvariableop_resource:Y
Gmodel_2_simple_rnn_6_simple_rnn_cell_6_matmul_1_readvariableop_resource:;
)dense_2_tensordot_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:
identityЂdense_2/BiasAdd/ReadVariableOpЂ dense_2/Tensordot/ReadVariableOpЂ=model_2/simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOpЂ<model_2/simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOpЂ>model_2/simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOpЂmodel_2/simple_rnn_5/whileЂ=model_2/simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOpЂ<model_2/simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOpЂ>model_2/simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOpЂmodel_2/simple_rnn_6/whileP
model_2/simple_rnn_5/ShapeShapeinputs*
T0*
_output_shapes
:r
(model_2/simple_rnn_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*model_2/simple_rnn_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*model_2/simple_rnn_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:К
"model_2/simple_rnn_5/strided_sliceStridedSlice#model_2/simple_rnn_5/Shape:output:01model_2/simple_rnn_5/strided_slice/stack:output:03model_2/simple_rnn_5/strided_slice/stack_1:output:03model_2/simple_rnn_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#model_2/simple_rnn_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
В
!model_2/simple_rnn_5/zeros/packedPack+model_2/simple_rnn_5/strided_slice:output:0,model_2/simple_rnn_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:e
 model_2/simple_rnn_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ћ
model_2/simple_rnn_5/zerosFill*model_2/simple_rnn_5/zeros/packed:output:0)model_2/simple_rnn_5/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
x
#model_2/simple_rnn_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
model_2/simple_rnn_5/transpose	Transposeinputs,model_2/simple_rnn_5/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџn
model_2/simple_rnn_5/Shape_1Shape"model_2/simple_rnn_5/transpose:y:0*
T0*
_output_shapes
:t
*model_2/simple_rnn_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,model_2/simple_rnn_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,model_2/simple_rnn_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ф
$model_2/simple_rnn_5/strided_slice_1StridedSlice%model_2/simple_rnn_5/Shape_1:output:03model_2/simple_rnn_5/strided_slice_1/stack:output:05model_2/simple_rnn_5/strided_slice_1/stack_1:output:05model_2/simple_rnn_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
0model_2/simple_rnn_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџѓ
"model_2/simple_rnn_5/TensorArrayV2TensorListReserve9model_2/simple_rnn_5/TensorArrayV2/element_shape:output:0-model_2/simple_rnn_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
Jmodel_2/simple_rnn_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
<model_2/simple_rnn_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"model_2/simple_rnn_5/transpose:y:0Smodel_2/simple_rnn_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвt
*model_2/simple_rnn_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,model_2/simple_rnn_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,model_2/simple_rnn_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:в
$model_2/simple_rnn_5/strided_slice_2StridedSlice"model_2/simple_rnn_5/transpose:y:03model_2/simple_rnn_5/strided_slice_2/stack:output:05model_2/simple_rnn_5/strided_slice_2/stack_1:output:05model_2/simple_rnn_5/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskТ
<model_2/simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOpEmodel_2_simple_rnn_5_simple_rnn_cell_5_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0о
-model_2/simple_rnn_5/simple_rnn_cell_5/MatMulMatMul-model_2/simple_rnn_5/strided_slice_2:output:0Dmodel_2/simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Р
=model_2/simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOpFmodel_2_simple_rnn_5_simple_rnn_cell_5_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0ы
.model_2/simple_rnn_5/simple_rnn_cell_5/BiasAddBiasAdd7model_2/simple_rnn_5/simple_rnn_cell_5/MatMul:product:0Emodel_2/simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Ц
>model_2/simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOpGmodel_2_simple_rnn_5_simple_rnn_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype0и
/model_2/simple_rnn_5/simple_rnn_cell_5/MatMul_1MatMul#model_2/simple_rnn_5/zeros:output:0Fmodel_2/simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
й
*model_2/simple_rnn_5/simple_rnn_cell_5/addAddV27model_2/simple_rnn_5/simple_rnn_cell_5/BiasAdd:output:09model_2/simple_rnn_5/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ

+model_2/simple_rnn_5/simple_rnn_cell_5/ReluRelu.model_2/simple_rnn_5/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ

2model_2/simple_rnn_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   s
1model_2/simple_rnn_5/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
$model_2/simple_rnn_5/TensorArrayV2_1TensorListReserve;model_2/simple_rnn_5/TensorArrayV2_1/element_shape:output:0:model_2/simple_rnn_5/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв[
model_2/simple_rnn_5/timeConst*
_output_shapes
: *
dtype0*
value	B : x
-model_2/simple_rnn_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџi
'model_2/simple_rnn_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : щ
model_2/simple_rnn_5/whileWhile0model_2/simple_rnn_5/while/loop_counter:output:06model_2/simple_rnn_5/while/maximum_iterations:output:0"model_2/simple_rnn_5/time:output:0-model_2/simple_rnn_5/TensorArrayV2_1:handle:0#model_2/simple_rnn_5/zeros:output:0-model_2/simple_rnn_5/strided_slice_1:output:0Lmodel_2/simple_rnn_5/TensorArrayUnstack/TensorListFromTensor:output_handle:0Emodel_2_simple_rnn_5_simple_rnn_cell_5_matmul_readvariableop_resourceFmodel_2_simple_rnn_5_simple_rnn_cell_5_biasadd_readvariableop_resourceGmodel_2_simple_rnn_5_simple_rnn_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *2
body*R(
&model_2_simple_rnn_5_while_body_240352*2
cond*R(
&model_2_simple_rnn_5_while_cond_240351*8
output_shapes'
%: : : : :џџџџџџџџџ
: : : : : *
parallel_iterations 
Emodel_2/simple_rnn_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   
7model_2/simple_rnn_5/TensorArrayV2Stack/TensorListStackTensorListStack#model_2/simple_rnn_5/while:output:3Nmodel_2/simple_rnn_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ
*
element_dtype0*
num_elements}
*model_2/simple_rnn_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџv
,model_2/simple_rnn_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,model_2/simple_rnn_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:№
$model_2/simple_rnn_5/strided_slice_3StridedSlice@model_2/simple_rnn_5/TensorArrayV2Stack/TensorListStack:tensor:03model_2/simple_rnn_5/strided_slice_3/stack:output:05model_2/simple_rnn_5/strided_slice_3/stack_1:output:05model_2/simple_rnn_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ
*
shrink_axis_maskz
%model_2/simple_rnn_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          е
 model_2/simple_rnn_5/transpose_1	Transpose@model_2/simple_rnn_5/TensorArrayV2Stack/TensorListStack:tensor:0.model_2/simple_rnn_5/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
h
&model_2/repeat_vector_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
"model_2/repeat_vector_2/ExpandDims
ExpandDims-model_2/simple_rnn_5/strided_slice_3:output:0/model_2/repeat_vector_2/ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
r
model_2/repeat_vector_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"         Џ
model_2/repeat_vector_2/TileTile+model_2/repeat_vector_2/ExpandDims:output:0&model_2/repeat_vector_2/stack:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
o
model_2/simple_rnn_6/ShapeShape%model_2/repeat_vector_2/Tile:output:0*
T0*
_output_shapes
:r
(model_2/simple_rnn_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*model_2/simple_rnn_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*model_2/simple_rnn_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:К
"model_2/simple_rnn_6/strided_sliceStridedSlice#model_2/simple_rnn_6/Shape:output:01model_2/simple_rnn_6/strided_slice/stack:output:03model_2/simple_rnn_6/strided_slice/stack_1:output:03model_2/simple_rnn_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#model_2/simple_rnn_6/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :В
!model_2/simple_rnn_6/zeros/packedPack+model_2/simple_rnn_6/strided_slice:output:0,model_2/simple_rnn_6/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:e
 model_2/simple_rnn_6/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ћ
model_2/simple_rnn_6/zerosFill*model_2/simple_rnn_6/zeros/packed:output:0)model_2/simple_rnn_6/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџx
#model_2/simple_rnn_6/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ж
model_2/simple_rnn_6/transpose	Transpose%model_2/repeat_vector_2/Tile:output:0,model_2/simple_rnn_6/transpose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
n
model_2/simple_rnn_6/Shape_1Shape"model_2/simple_rnn_6/transpose:y:0*
T0*
_output_shapes
:t
*model_2/simple_rnn_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,model_2/simple_rnn_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,model_2/simple_rnn_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ф
$model_2/simple_rnn_6/strided_slice_1StridedSlice%model_2/simple_rnn_6/Shape_1:output:03model_2/simple_rnn_6/strided_slice_1/stack:output:05model_2/simple_rnn_6/strided_slice_1/stack_1:output:05model_2/simple_rnn_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
0model_2/simple_rnn_6/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџѓ
"model_2/simple_rnn_6/TensorArrayV2TensorListReserve9model_2/simple_rnn_6/TensorArrayV2/element_shape:output:0-model_2/simple_rnn_6/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
Jmodel_2/simple_rnn_6/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   
<model_2/simple_rnn_6/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"model_2/simple_rnn_6/transpose:y:0Smodel_2/simple_rnn_6/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвt
*model_2/simple_rnn_6/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,model_2/simple_rnn_6/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,model_2/simple_rnn_6/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:в
$model_2/simple_rnn_6/strided_slice_2StridedSlice"model_2/simple_rnn_6/transpose:y:03model_2/simple_rnn_6/strided_slice_2/stack:output:05model_2/simple_rnn_6/strided_slice_2/stack_1:output:05model_2/simple_rnn_6/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ
*
shrink_axis_maskТ
<model_2/simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOpReadVariableOpEmodel_2_simple_rnn_6_simple_rnn_cell_6_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0о
-model_2/simple_rnn_6/simple_rnn_cell_6/MatMulMatMul-model_2/simple_rnn_6/strided_slice_2:output:0Dmodel_2/simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџР
=model_2/simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOpReadVariableOpFmodel_2_simple_rnn_6_simple_rnn_cell_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ы
.model_2/simple_rnn_6/simple_rnn_cell_6/BiasAddBiasAdd7model_2/simple_rnn_6/simple_rnn_cell_6/MatMul:product:0Emodel_2/simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЦ
>model_2/simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOpReadVariableOpGmodel_2_simple_rnn_6_simple_rnn_cell_6_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0и
/model_2/simple_rnn_6/simple_rnn_cell_6/MatMul_1MatMul#model_2/simple_rnn_6/zeros:output:0Fmodel_2/simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџй
*model_2/simple_rnn_6/simple_rnn_cell_6/addAddV27model_2/simple_rnn_6/simple_rnn_cell_6/BiasAdd:output:09model_2/simple_rnn_6/simple_rnn_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ
+model_2/simple_rnn_6/simple_rnn_cell_6/ReluRelu.model_2/simple_rnn_6/simple_rnn_cell_6/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
2model_2/simple_rnn_6/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ї
$model_2/simple_rnn_6/TensorArrayV2_1TensorListReserve;model_2/simple_rnn_6/TensorArrayV2_1/element_shape:output:0-model_2/simple_rnn_6/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв[
model_2/simple_rnn_6/timeConst*
_output_shapes
: *
dtype0*
value	B : x
-model_2/simple_rnn_6/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџi
'model_2/simple_rnn_6/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : щ
model_2/simple_rnn_6/whileWhile0model_2/simple_rnn_6/while/loop_counter:output:06model_2/simple_rnn_6/while/maximum_iterations:output:0"model_2/simple_rnn_6/time:output:0-model_2/simple_rnn_6/TensorArrayV2_1:handle:0#model_2/simple_rnn_6/zeros:output:0-model_2/simple_rnn_6/strided_slice_1:output:0Lmodel_2/simple_rnn_6/TensorArrayUnstack/TensorListFromTensor:output_handle:0Emodel_2_simple_rnn_6_simple_rnn_cell_6_matmul_readvariableop_resourceFmodel_2_simple_rnn_6_simple_rnn_cell_6_biasadd_readvariableop_resourceGmodel_2_simple_rnn_6_simple_rnn_cell_6_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *2
body*R(
&model_2_simple_rnn_6_while_body_240461*2
cond*R(
&model_2_simple_rnn_6_while_cond_240460*8
output_shapes'
%: : : : :џџџџџџџџџ: : : : : *
parallel_iterations 
Emodel_2/simple_rnn_6/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
7model_2/simple_rnn_6/TensorArrayV2Stack/TensorListStackTensorListStack#model_2/simple_rnn_6/while:output:3Nmodel_2/simple_rnn_6/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ*
element_dtype0}
*model_2/simple_rnn_6/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџv
,model_2/simple_rnn_6/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,model_2/simple_rnn_6/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:№
$model_2/simple_rnn_6/strided_slice_3StridedSlice@model_2/simple_rnn_6/TensorArrayV2Stack/TensorListStack:tensor:03model_2/simple_rnn_6/strided_slice_3/stack:output:05model_2/simple_rnn_6/strided_slice_3/stack_1:output:05model_2/simple_rnn_6/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskz
%model_2/simple_rnn_6/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          е
 model_2/simple_rnn_6/transpose_1	Transpose@model_2/simple_rnn_6/TensorArrayV2Stack/TensorListStack:tensor:0.model_2/simple_rnn_6/transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0`
dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       k
dense_2/Tensordot/ShapeShape$model_2/simple_rnn_6/transpose_1:y:0*
T0*
_output_shapes
:a
dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : л
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : п
dense_2/Tensordot/GatherV2_1GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/axes:output:0*dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : М
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ї
dense_2/Tensordot/transpose	Transpose$model_2/simple_rnn_6/transpose_1:y:0!dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџЂ
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЂ
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџc
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:a
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ч
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџk
IdentityIdentitydense_2/BiasAdd:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџФ
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp>^model_2/simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp=^model_2/simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp?^model_2/simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp^model_2/simple_rnn_5/while>^model_2/simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOp=^model_2/simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOp?^model_2/simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOp^model_2/simple_rnn_6/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':џџџџџџџџџ: : : : : : : : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2~
=model_2/simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp=model_2/simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp2|
<model_2/simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp<model_2/simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp2
>model_2/simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp>model_2/simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp28
model_2/simple_rnn_5/whilemodel_2/simple_rnn_5/while2~
=model_2/simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOp=model_2/simple_rnn_6/simple_rnn_cell_6/BiasAdd/ReadVariableOp2|
<model_2/simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOp<model_2/simple_rnn_6/simple_rnn_cell_6/MatMul/ReadVariableOp2
>model_2/simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOp>model_2/simple_rnn_6/simple_rnn_cell_6/MatMul_1/ReadVariableOp28
model_2/simple_rnn_6/whilemodel_2/simple_rnn_6/while:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ЁL
Ш
3sequential_5_model_2_simple_rnn_6_while_body_238458`
\sequential_5_model_2_simple_rnn_6_while_sequential_5_model_2_simple_rnn_6_while_loop_counterf
bsequential_5_model_2_simple_rnn_6_while_sequential_5_model_2_simple_rnn_6_while_maximum_iterations7
3sequential_5_model_2_simple_rnn_6_while_placeholder9
5sequential_5_model_2_simple_rnn_6_while_placeholder_19
5sequential_5_model_2_simple_rnn_6_while_placeholder_2_
[sequential_5_model_2_simple_rnn_6_while_sequential_5_model_2_simple_rnn_6_strided_slice_1_0
sequential_5_model_2_simple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_sequential_5_model_2_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor_0l
Zsequential_5_model_2_simple_rnn_6_while_simple_rnn_cell_6_matmul_readvariableop_resource_0:
i
[sequential_5_model_2_simple_rnn_6_while_simple_rnn_cell_6_biasadd_readvariableop_resource_0:n
\sequential_5_model_2_simple_rnn_6_while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0:4
0sequential_5_model_2_simple_rnn_6_while_identity6
2sequential_5_model_2_simple_rnn_6_while_identity_16
2sequential_5_model_2_simple_rnn_6_while_identity_26
2sequential_5_model_2_simple_rnn_6_while_identity_36
2sequential_5_model_2_simple_rnn_6_while_identity_4]
Ysequential_5_model_2_simple_rnn_6_while_sequential_5_model_2_simple_rnn_6_strided_slice_1
sequential_5_model_2_simple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_sequential_5_model_2_simple_rnn_6_tensorarrayunstack_tensorlistfromtensorj
Xsequential_5_model_2_simple_rnn_6_while_simple_rnn_cell_6_matmul_readvariableop_resource:
g
Ysequential_5_model_2_simple_rnn_6_while_simple_rnn_cell_6_biasadd_readvariableop_resource:l
Zsequential_5_model_2_simple_rnn_6_while_simple_rnn_cell_6_matmul_1_readvariableop_resource:ЂPsequential_5/model_2/simple_rnn_6/while/simple_rnn_cell_6/BiasAdd/ReadVariableOpЂOsequential_5/model_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul/ReadVariableOpЂQsequential_5/model_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul_1/ReadVariableOpЊ
Ysequential_5/model_2/simple_rnn_6/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   б
Ksequential_5/model_2/simple_rnn_6/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_5_model_2_simple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_sequential_5_model_2_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor_03sequential_5_model_2_simple_rnn_6_while_placeholderbsequential_5/model_2/simple_rnn_6/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:џџџџџџџџџ
*
element_dtype0ъ
Osequential_5/model_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul/ReadVariableOpReadVariableOpZsequential_5_model_2_simple_rnn_6_while_simple_rnn_cell_6_matmul_readvariableop_resource_0*
_output_shapes

:
*
dtype0Љ
@sequential_5/model_2/simple_rnn_6/while/simple_rnn_cell_6/MatMulMatMulRsequential_5/model_2/simple_rnn_6/while/TensorArrayV2Read/TensorListGetItem:item:0Wsequential_5/model_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџш
Psequential_5/model_2/simple_rnn_6/while/simple_rnn_cell_6/BiasAdd/ReadVariableOpReadVariableOp[sequential_5_model_2_simple_rnn_6_while_simple_rnn_cell_6_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0Є
Asequential_5/model_2/simple_rnn_6/while/simple_rnn_cell_6/BiasAddBiasAddJsequential_5/model_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul:product:0Xsequential_5/model_2/simple_rnn_6/while/simple_rnn_cell_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџю
Qsequential_5/model_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul_1/ReadVariableOpReadVariableOp\sequential_5_model_2_simple_rnn_6_while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0
Bsequential_5/model_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul_1MatMul5sequential_5_model_2_simple_rnn_6_while_placeholder_2Ysequential_5/model_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
=sequential_5/model_2/simple_rnn_6/while/simple_rnn_cell_6/addAddV2Jsequential_5/model_2/simple_rnn_6/while/simple_rnn_cell_6/BiasAdd:output:0Lsequential_5/model_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџЛ
>sequential_5/model_2/simple_rnn_6/while/simple_rnn_cell_6/ReluReluAsequential_5/model_2/simple_rnn_6/while/simple_rnn_cell_6/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџл
Lsequential_5/model_2/simple_rnn_6/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem5sequential_5_model_2_simple_rnn_6_while_placeholder_13sequential_5_model_2_simple_rnn_6_while_placeholderLsequential_5/model_2/simple_rnn_6/while/simple_rnn_cell_6/Relu:activations:0*
_output_shapes
: *
element_dtype0:щшвo
-sequential_5/model_2/simple_rnn_6/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Т
+sequential_5/model_2/simple_rnn_6/while/addAddV23sequential_5_model_2_simple_rnn_6_while_placeholder6sequential_5/model_2/simple_rnn_6/while/add/y:output:0*
T0*
_output_shapes
: q
/sequential_5/model_2/simple_rnn_6/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :я
-sequential_5/model_2/simple_rnn_6/while/add_1AddV2\sequential_5_model_2_simple_rnn_6_while_sequential_5_model_2_simple_rnn_6_while_loop_counter8sequential_5/model_2/simple_rnn_6/while/add_1/y:output:0*
T0*
_output_shapes
: П
0sequential_5/model_2/simple_rnn_6/while/IdentityIdentity1sequential_5/model_2/simple_rnn_6/while/add_1:z:0-^sequential_5/model_2/simple_rnn_6/while/NoOp*
T0*
_output_shapes
: ђ
2sequential_5/model_2/simple_rnn_6/while/Identity_1Identitybsequential_5_model_2_simple_rnn_6_while_sequential_5_model_2_simple_rnn_6_while_maximum_iterations-^sequential_5/model_2/simple_rnn_6/while/NoOp*
T0*
_output_shapes
: П
2sequential_5/model_2/simple_rnn_6/while/Identity_2Identity/sequential_5/model_2/simple_rnn_6/while/add:z:0-^sequential_5/model_2/simple_rnn_6/while/NoOp*
T0*
_output_shapes
: ь
2sequential_5/model_2/simple_rnn_6/while/Identity_3Identity\sequential_5/model_2/simple_rnn_6/while/TensorArrayV2Write/TensorListSetItem:output_handle:0-^sequential_5/model_2/simple_rnn_6/while/NoOp*
T0*
_output_shapes
: э
2sequential_5/model_2/simple_rnn_6/while/Identity_4IdentityLsequential_5/model_2/simple_rnn_6/while/simple_rnn_cell_6/Relu:activations:0-^sequential_5/model_2/simple_rnn_6/while/NoOp*
T0*'
_output_shapes
:џџџџџџџџџч
,sequential_5/model_2/simple_rnn_6/while/NoOpNoOpQ^sequential_5/model_2/simple_rnn_6/while/simple_rnn_cell_6/BiasAdd/ReadVariableOpP^sequential_5/model_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul/ReadVariableOpR^sequential_5/model_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "m
0sequential_5_model_2_simple_rnn_6_while_identity9sequential_5/model_2/simple_rnn_6/while/Identity:output:0"q
2sequential_5_model_2_simple_rnn_6_while_identity_1;sequential_5/model_2/simple_rnn_6/while/Identity_1:output:0"q
2sequential_5_model_2_simple_rnn_6_while_identity_2;sequential_5/model_2/simple_rnn_6/while/Identity_2:output:0"q
2sequential_5_model_2_simple_rnn_6_while_identity_3;sequential_5/model_2/simple_rnn_6/while/Identity_3:output:0"q
2sequential_5_model_2_simple_rnn_6_while_identity_4;sequential_5/model_2/simple_rnn_6/while/Identity_4:output:0"И
Ysequential_5_model_2_simple_rnn_6_while_sequential_5_model_2_simple_rnn_6_strided_slice_1[sequential_5_model_2_simple_rnn_6_while_sequential_5_model_2_simple_rnn_6_strided_slice_1_0"И
Ysequential_5_model_2_simple_rnn_6_while_simple_rnn_cell_6_biasadd_readvariableop_resource[sequential_5_model_2_simple_rnn_6_while_simple_rnn_cell_6_biasadd_readvariableop_resource_0"К
Zsequential_5_model_2_simple_rnn_6_while_simple_rnn_cell_6_matmul_1_readvariableop_resource\sequential_5_model_2_simple_rnn_6_while_simple_rnn_cell_6_matmul_1_readvariableop_resource_0"Ж
Xsequential_5_model_2_simple_rnn_6_while_simple_rnn_cell_6_matmul_readvariableop_resourceZsequential_5_model_2_simple_rnn_6_while_simple_rnn_cell_6_matmul_readvariableop_resource_0"В
sequential_5_model_2_simple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_sequential_5_model_2_simple_rnn_6_tensorarrayunstack_tensorlistfromtensorsequential_5_model_2_simple_rnn_6_while_tensorarrayv2read_tensorlistgetitem_sequential_5_model_2_simple_rnn_6_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :џџџџџџџџџ: : : : : 2Є
Psequential_5/model_2/simple_rnn_6/while/simple_rnn_cell_6/BiasAdd/ReadVariableOpPsequential_5/model_2/simple_rnn_6/while/simple_rnn_cell_6/BiasAdd/ReadVariableOp2Ђ
Osequential_5/model_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul/ReadVariableOpOsequential_5/model_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul/ReadVariableOp2І
Qsequential_5/model_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul_1/ReadVariableOpQsequential_5/model_2/simple_rnn_6/while/simple_rnn_cell_6/MatMul_1/ReadVariableOp: 
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
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
Ј>
Л
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_241546

inputsB
0simple_rnn_cell_5_matmul_readvariableop_resource:
?
1simple_rnn_cell_5_biasadd_readvariableop_resource:
D
2simple_rnn_cell_5_matmul_1_readvariableop_resource:


identityЂ(simple_rnn_cell_5/BiasAdd/ReadVariableOpЂ'simple_rnn_cell_5/MatMul/ReadVariableOpЂ)simple_rnn_cell_5/MatMul_1/ReadVariableOpЂwhile;
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
value	B :
s
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
:џџџџџџџџџ
c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџD
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
valueB"џџџџ   р
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
valueB:щ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
'simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_5_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0
simple_rnn_cell_5/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ

(simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_5_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Ќ
simple_rnn_cell_5/BiasAddBiasAdd"simple_rnn_cell_5/MatMul:product:00simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ

)simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:

*
dtype0
simple_rnn_cell_5/MatMul_1MatMulzeros:output:01simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ

simple_rnn_cell_5/addAddV2"simple_rnn_cell_5/BiasAdd:output:0$simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџ
k
simple_rnn_cell_5/ReluRelusimple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   ^
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
value	B : и
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_5_matmul_readvariableop_resource1simple_rnn_cell_5_biasadd_readvariableop_resource2simple_rnn_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :џџџџџџџџџ
: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_241479*
condR
while_cond_241478*8
output_shapes'
%: : : : :џџџџџџџџџ
: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ
   ж
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:џџџџџџџџџ
*
element_dtype0*
num_elementsh
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
:џџџџџџџџџ
*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
Я
NoOpNoOp)^simple_rnn_cell_5/BiasAdd/ReadVariableOp(^simple_rnn_cell_5/MatMul/ReadVariableOp*^simple_rnn_cell_5/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:џџџџџџџџџ: : : 2T
(simple_rnn_cell_5/BiasAdd/ReadVariableOp(simple_rnn_cell_5/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_5/MatMul/ReadVariableOp'simple_rnn_cell_5/MatMul/ReadVariableOp2V
)simple_rnn_cell_5/MatMul_1/ReadVariableOp)simple_rnn_cell_5/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ђ
А
3sequential_5_model_2_simple_rnn_5_while_cond_238348`
\sequential_5_model_2_simple_rnn_5_while_sequential_5_model_2_simple_rnn_5_while_loop_counterf
bsequential_5_model_2_simple_rnn_5_while_sequential_5_model_2_simple_rnn_5_while_maximum_iterations7
3sequential_5_model_2_simple_rnn_5_while_placeholder9
5sequential_5_model_2_simple_rnn_5_while_placeholder_19
5sequential_5_model_2_simple_rnn_5_while_placeholder_2b
^sequential_5_model_2_simple_rnn_5_while_less_sequential_5_model_2_simple_rnn_5_strided_slice_1x
tsequential_5_model_2_simple_rnn_5_while_sequential_5_model_2_simple_rnn_5_while_cond_238348___redundant_placeholder0x
tsequential_5_model_2_simple_rnn_5_while_sequential_5_model_2_simple_rnn_5_while_cond_238348___redundant_placeholder1x
tsequential_5_model_2_simple_rnn_5_while_sequential_5_model_2_simple_rnn_5_while_cond_238348___redundant_placeholder2x
tsequential_5_model_2_simple_rnn_5_while_sequential_5_model_2_simple_rnn_5_while_cond_238348___redundant_placeholder34
0sequential_5_model_2_simple_rnn_5_while_identity
ъ
,sequential_5/model_2/simple_rnn_5/while/LessLess3sequential_5_model_2_simple_rnn_5_while_placeholder^sequential_5_model_2_simple_rnn_5_while_less_sequential_5_model_2_simple_rnn_5_strided_slice_1*
T0*
_output_shapes
: 
0sequential_5/model_2/simple_rnn_5/while/IdentityIdentity0sequential_5/model_2/simple_rnn_5/while/Less:z:0*
T0
*
_output_shapes
: "m
0sequential_5_model_2_simple_rnn_5_while_identity9sequential_5/model_2/simple_rnn_5/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :џџџџџџџџџ
: ::::: 
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
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
:
а

(__inference_dense_2_layer_call_fn_241032

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_239829s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ`
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
к
Њ
while_cond_239201
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_239201___redundant_placeholder04
0while_while_cond_239201___redundant_placeholder14
0while_while_cond_239201___redundant_placeholder24
0while_while_cond_239201___redundant_placeholder3
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
-: : : : :џџџџџџџџџ
: ::::: 
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
:џџџџџџџџџ
:

_output_shapes
: :

_output_shapes
:
Й

к
2__inference_simple_rnn_cell_5_layer_call_fn_242063

inputs
states_0
unknown:

	unknown_0:

	unknown_1:


identity

identity_1ЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ
:џџџџџџџџџ
*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_238720o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:џџџџџџџџџ:џџџџџџџџџ
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ

"
_user_specified_name
states/0"Е	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*О
serving_defaultЊ
K
model_2_input:
serving_default_model_2_input:0џџџџџџџџџ?
dense_24
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:іЅ
Д
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	_default_save_signature

	optimizer

signatures"
_tf_keras_sequential

layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_network
Л
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
X
0
1
 2
!3
"4
#5
6
7"
trackable_list_wrapper
X
0
1
 2
!3
"4
#5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
$non_trainable_variables

%layers
&metrics
'layer_regularization_losses
(layer_metrics
	variables
trainable_variables
regularization_losses
__call__
	_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
щ
)trace_0
*trace_1
+trace_2
,trace_32ў
-__inference_sequential_5_layer_call_fn_239855
-__inference_sequential_5_layer_call_fn_240044
-__inference_sequential_5_layer_call_fn_240065
-__inference_sequential_5_layer_call_fn_239950П
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
 z)trace_0z*trace_1z+trace_2z,trace_3
е
-trace_0
.trace_1
/trace_2
0trace_32ъ
H__inference_sequential_5_layer_call_and_return_conditional_losses_240309
H__inference_sequential_5_layer_call_and_return_conditional_losses_240553
H__inference_sequential_5_layer_call_and_return_conditional_losses_239972
H__inference_sequential_5_layer_call_and_return_conditional_losses_239994П
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
 z-trace_0z.trace_1z/trace_2z0trace_3
вBЯ
!__inference__wrapped_model_238550model_2_input"
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
ѓ
1iter

2beta_1

3beta_2
	4decay
5learning_ratemЎmЏmАmБ mВ!mГ"mД#mЕvЖvЗvИvЙ vК!vЛ"vМ#vН"
	optimizer
,
6serving_default"
signature_map
"
_tf_keras_input_layer
У
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses
=cell
>
state_spec"
_tf_keras_rnn_layer
Ѕ
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses"
_tf_keras_layer
У
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses
Kcell
L
state_spec"
_tf_keras_rnn_layer
J
0
1
 2
!3
"4
#5"
trackable_list_wrapper
J
0
1
 2
!3
"4
#5"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
е
Rtrace_0
Strace_1
Ttrace_2
Utrace_32ъ
(__inference_model_2_layer_call_fn_239409
(__inference_model_2_layer_call_fn_240570
(__inference_model_2_layer_call_fn_240587
(__inference_model_2_layer_call_fn_239741П
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
С
Vtrace_0
Wtrace_1
Xtrace_2
Ytrace_32ж
C__inference_model_2_layer_call_and_return_conditional_losses_240805
C__inference_model_2_layer_call_and_return_conditional_losses_241023
C__inference_model_2_layer_call_and_return_conditional_losses_239760
C__inference_model_2_layer_call_and_return_conditional_losses_239779П
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
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ь
_trace_02Я
(__inference_dense_2_layer_call_fn_241032Ђ
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
 z_trace_0

`trace_02ъ
C__inference_dense_2_layer_call_and_return_conditional_losses_241062Ђ
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
 z`trace_0
 :2dense_2/kernel
:2dense_2/bias
7:5
2%simple_rnn_5/simple_rnn_cell_5/kernel
A:?

2/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel
1:/
2#simple_rnn_5/simple_rnn_cell_5/bias
7:5
2%simple_rnn_6/simple_rnn_cell_6/kernel
A:?2/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel
1:/2#simple_rnn_6/simple_rnn_cell_6/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
-__inference_sequential_5_layer_call_fn_239855model_2_input"П
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
ўBћ
-__inference_sequential_5_layer_call_fn_240044inputs"П
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
ўBћ
-__inference_sequential_5_layer_call_fn_240065inputs"П
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
B
-__inference_sequential_5_layer_call_fn_239950model_2_input"П
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
B
H__inference_sequential_5_layer_call_and_return_conditional_losses_240309inputs"П
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
B
H__inference_sequential_5_layer_call_and_return_conditional_losses_240553inputs"П
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
 B
H__inference_sequential_5_layer_call_and_return_conditional_losses_239972model_2_input"П
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
 B
H__inference_sequential_5_layer_call_and_return_conditional_losses_239994model_2_input"П
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
бBЮ
$__inference_signature_wrapper_240023model_2_input"
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
5
0
1
 2"
trackable_list_wrapper
5
0
1
 2"
trackable_list_wrapper
 "
trackable_list_wrapper
Й

cstates
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
ў
itrace_0
jtrace_1
ktrace_2
ltrace_32
-__inference_simple_rnn_5_layer_call_fn_241073
-__inference_simple_rnn_5_layer_call_fn_241084
-__inference_simple_rnn_5_layer_call_fn_241095
-__inference_simple_rnn_5_layer_call_fn_241106д
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
 zitrace_0zjtrace_1zktrace_2zltrace_3
ъ
mtrace_0
ntrace_1
otrace_2
ptrace_32џ
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_241216
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_241326
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_241436
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_241546д
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
 zmtrace_0zntrace_1zotrace_2zptrace_3
ш
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses
w_random_generator

kernel
recurrent_kernel
 bias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
є
}trace_02з
0__inference_repeat_vector_2_layer_call_fn_241551Ђ
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
 z}trace_0

~trace_02ђ
K__inference_repeat_vector_2_layer_call_and_return_conditional_losses_241559Ђ
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
 z~trace_0
5
!0
"1
#2"
trackable_list_wrapper
5
!0
"1
#2"
trackable_list_wrapper
 "
trackable_list_wrapper
О

states
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object

trace_0
trace_1
trace_2
trace_32
-__inference_simple_rnn_6_layer_call_fn_241570
-__inference_simple_rnn_6_layer_call_fn_241581
-__inference_simple_rnn_6_layer_call_fn_241592
-__inference_simple_rnn_6_layer_call_fn_241603д
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
 ztrace_0ztrace_1ztrace_2ztrace_3
ђ
trace_0
trace_1
trace_2
trace_32џ
H__inference_simple_rnn_6_layer_call_and_return_conditional_losses_241711
H__inference_simple_rnn_6_layer_call_and_return_conditional_losses_241819
H__inference_simple_rnn_6_layer_call_and_return_conditional_losses_241927
H__inference_simple_rnn_6_layer_call_and_return_conditional_losses_242035д
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
 ztrace_0ztrace_1ztrace_2ztrace_3
я
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator

!kernel
"recurrent_kernel
#bias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
њBї
(__inference_model_2_layer_call_fn_239409input_6"П
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
љBі
(__inference_model_2_layer_call_fn_240570inputs"П
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
љBі
(__inference_model_2_layer_call_fn_240587inputs"П
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
њBї
(__inference_model_2_layer_call_fn_239741input_6"П
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
B
C__inference_model_2_layer_call_and_return_conditional_losses_240805inputs"П
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
B
C__inference_model_2_layer_call_and_return_conditional_losses_241023inputs"П
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
B
C__inference_model_2_layer_call_and_return_conditional_losses_239760input_6"П
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
B
C__inference_model_2_layer_call_and_return_conditional_losses_239779input_6"П
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
мBй
(__inference_dense_2_layer_call_fn_241032inputs"Ђ
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
їBє
C__inference_dense_2_layer_call_and_return_conditional_losses_241062inputs"Ђ
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
	variables
	keras_api

total

count"
_tf_keras_metric
R
	variables
	keras_api

total

count"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
=0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
-__inference_simple_rnn_5_layer_call_fn_241073inputs/0"д
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
B
-__inference_simple_rnn_5_layer_call_fn_241084inputs/0"д
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
B
-__inference_simple_rnn_5_layer_call_fn_241095inputs"д
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
B
-__inference_simple_rnn_5_layer_call_fn_241106inputs"д
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
АB­
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_241216inputs/0"д
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
АB­
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_241326inputs/0"д
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
ЎBЋ
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_241436inputs"д
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
ЎBЋ
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_241546inputs"д
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
0
1
 2"
trackable_list_wrapper
5
0
1
 2"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
 layer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
у
Ёtrace_0
Ђtrace_12Ј
2__inference_simple_rnn_cell_5_layer_call_fn_242049
2__inference_simple_rnn_cell_5_layer_call_fn_242063Н
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
 zЁtrace_0zЂtrace_1

Ѓtrace_0
Єtrace_12о
M__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_242080
M__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_242097Н
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
 zЃtrace_0zЄtrace_1
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
фBс
0__inference_repeat_vector_2_layer_call_fn_241551inputs"Ђ
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
џBќ
K__inference_repeat_vector_2_layer_call_and_return_conditional_losses_241559inputs"Ђ
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
K0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
-__inference_simple_rnn_6_layer_call_fn_241570inputs/0"д
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
B
-__inference_simple_rnn_6_layer_call_fn_241581inputs/0"д
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
B
-__inference_simple_rnn_6_layer_call_fn_241592inputs"д
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
B
-__inference_simple_rnn_6_layer_call_fn_241603inputs"д
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
АB­
H__inference_simple_rnn_6_layer_call_and_return_conditional_losses_241711inputs/0"д
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
АB­
H__inference_simple_rnn_6_layer_call_and_return_conditional_losses_241819inputs/0"д
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
ЎBЋ
H__inference_simple_rnn_6_layer_call_and_return_conditional_losses_241927inputs"д
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
ЎBЋ
H__inference_simple_rnn_6_layer_call_and_return_conditional_losses_242035inputs"д
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
!0
"1
#2"
trackable_list_wrapper
5
!0
"1
#2"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ѕnon_trainable_variables
Іlayers
Їmetrics
 Јlayer_regularization_losses
Љlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
у
Њtrace_0
Ћtrace_12Ј
2__inference_simple_rnn_cell_6_layer_call_fn_242111
2__inference_simple_rnn_cell_6_layer_call_fn_242125Н
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
 zЊtrace_0zЋtrace_1

Ќtrace_0
­trace_12о
M__inference_simple_rnn_cell_6_layer_call_and_return_conditional_losses_242142
M__inference_simple_rnn_cell_6_layer_call_and_return_conditional_losses_242159Н
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
 zЌtrace_0z­trace_1
"
_generic_user_object
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
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
B
2__inference_simple_rnn_cell_5_layer_call_fn_242049inputsstates/0"Н
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
B
2__inference_simple_rnn_cell_5_layer_call_fn_242063inputsstates/0"Н
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
ІBЃ
M__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_242080inputsstates/0"Н
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
ІBЃ
M__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_242097inputsstates/0"Н
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
B
2__inference_simple_rnn_cell_6_layer_call_fn_242111inputsstates/0"Н
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
B
2__inference_simple_rnn_cell_6_layer_call_fn_242125inputsstates/0"Н
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
ІBЃ
M__inference_simple_rnn_cell_6_layer_call_and_return_conditional_losses_242142inputsstates/0"Н
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
ІBЃ
M__inference_simple_rnn_cell_6_layer_call_and_return_conditional_losses_242159inputsstates/0"Н
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
%:#2Adam/dense_2/kernel/m
:2Adam/dense_2/bias/m
<::
2,Adam/simple_rnn_5/simple_rnn_cell_5/kernel/m
F:D

26Adam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/m
6:4
2*Adam/simple_rnn_5/simple_rnn_cell_5/bias/m
<::
2,Adam/simple_rnn_6/simple_rnn_cell_6/kernel/m
F:D26Adam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/m
6:42*Adam/simple_rnn_6/simple_rnn_cell_6/bias/m
%:#2Adam/dense_2/kernel/v
:2Adam/dense_2/bias/v
<::
2,Adam/simple_rnn_5/simple_rnn_cell_5/kernel/v
F:D

26Adam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/v
6:4
2*Adam/simple_rnn_5/simple_rnn_cell_5/bias/v
<::
2,Adam/simple_rnn_6/simple_rnn_cell_6/kernel/v
F:D26Adam/simple_rnn_6/simple_rnn_cell_6/recurrent_kernel/v
6:42*Adam/simple_rnn_6/simple_rnn_cell_6/bias/vЂ
!__inference__wrapped_model_238550} !#":Ђ7
0Ђ-
+(
model_2_inputџџџџџџџџџ
Њ "5Њ2
0
dense_2%"
dense_2џџџџџџџџџЋ
C__inference_dense_2_layer_call_and_return_conditional_losses_241062d3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ ")Ђ&

0џџџџџџџџџ
 
(__inference_dense_2_layer_call_fn_241032W3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ "џџџџџџџџџИ
C__inference_model_2_layer_call_and_return_conditional_losses_239760q !#"<Ђ9
2Ђ/
%"
input_6џџџџџџџџџ
p 

 
Њ ")Ђ&

0џџџџџџџџџ
 И
C__inference_model_2_layer_call_and_return_conditional_losses_239779q !#"<Ђ9
2Ђ/
%"
input_6џџџџџџџџџ
p

 
Њ ")Ђ&

0џџџџџџџџџ
 З
C__inference_model_2_layer_call_and_return_conditional_losses_240805p !#";Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p 

 
Њ ")Ђ&

0џџџџџџџџџ
 З
C__inference_model_2_layer_call_and_return_conditional_losses_241023p !#";Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p

 
Њ ")Ђ&

0џџџџџџџџџ
 
(__inference_model_2_layer_call_fn_239409d !#"<Ђ9
2Ђ/
%"
input_6џџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
(__inference_model_2_layer_call_fn_239741d !#"<Ђ9
2Ђ/
%"
input_6џџџџџџџџџ
p

 
Њ "џџџџџџџџџ
(__inference_model_2_layer_call_fn_240570c !#";Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
(__inference_model_2_layer_call_fn_240587c !#";Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџН
K__inference_repeat_vector_2_layer_call_and_return_conditional_losses_241559n8Ђ5
.Ђ+
)&
inputsџџџџџџџџџџџџџџџџџџ
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџ
 
0__inference_repeat_vector_2_layer_call_fn_241551a8Ђ5
.Ђ+
)&
inputsџџџџџџџџџџџџџџџџџџ
Њ "%"џџџџџџџџџџџџџџџџџџХ
H__inference_sequential_5_layer_call_and_return_conditional_losses_239972y !#"BЂ?
8Ђ5
+(
model_2_inputџџџџџџџџџ
p 

 
Њ ")Ђ&

0џџџџџџџџџ
 Х
H__inference_sequential_5_layer_call_and_return_conditional_losses_239994y !#"BЂ?
8Ђ5
+(
model_2_inputџџџџџџџџџ
p

 
Њ ")Ђ&

0џџџџџџџџџ
 О
H__inference_sequential_5_layer_call_and_return_conditional_losses_240309r !#";Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p 

 
Њ ")Ђ&

0џџџџџџџџџ
 О
H__inference_sequential_5_layer_call_and_return_conditional_losses_240553r !#";Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p

 
Њ ")Ђ&

0џџџџџџџџџ
 
-__inference_sequential_5_layer_call_fn_239855l !#"BЂ?
8Ђ5
+(
model_2_inputџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
-__inference_sequential_5_layer_call_fn_239950l !#"BЂ?
8Ђ5
+(
model_2_inputџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
-__inference_sequential_5_layer_call_fn_240044e !#";Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
-__inference_sequential_5_layer_call_fn_240065e !#";Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџЗ
$__inference_signature_wrapper_240023 !#"KЂH
Ђ 
AЊ>
<
model_2_input+(
model_2_inputџџџџџџџџџ"5Њ2
0
dense_2%"
dense_2џџџџџџџџџЩ
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_241216} OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "%Ђ"

0џџџџџџџџџ

 Щ
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_241326} OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "%Ђ"

0џџџџџџџџџ

 Й
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_241436m ?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p 

 
Њ "%Ђ"

0џџџџџџџџџ

 Й
H__inference_simple_rnn_5_layer_call_and_return_conditional_losses_241546m ?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p

 
Њ "%Ђ"

0џџџџџџџџџ

 Ё
-__inference_simple_rnn_5_layer_call_fn_241073p OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџ
Ё
-__inference_simple_rnn_5_layer_call_fn_241084p OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџ

-__inference_simple_rnn_5_layer_call_fn_241095` ?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p 

 
Њ "џџџџџџџџџ

-__inference_simple_rnn_5_layer_call_fn_241106` ?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ

 
p

 
Њ "џџџџџџџџџ
з
H__inference_simple_rnn_6_layer_call_and_return_conditional_losses_241711!#"OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ


 
p 

 
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџ
 з
H__inference_simple_rnn_6_layer_call_and_return_conditional_losses_241819!#"OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ


 
p

 
Њ "2Ђ/
(%
0џџџџџџџџџџџџџџџџџџ
 Н
H__inference_simple_rnn_6_layer_call_and_return_conditional_losses_241927q!#"?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ


 
p 

 
Њ ")Ђ&

0џџџџџџџџџ
 Н
H__inference_simple_rnn_6_layer_call_and_return_conditional_losses_242035q!#"?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ


 
p

 
Њ ")Ђ&

0џџџџџџџџџ
 Ў
-__inference_simple_rnn_6_layer_call_fn_241570}!#"OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ


 
p 

 
Њ "%"џџџџџџџџџџџџџџџџџџЎ
-__inference_simple_rnn_6_layer_call_fn_241581}!#"OЂL
EЂB
41
/,
inputs/0џџџџџџџџџџџџџџџџџџ


 
p

 
Њ "%"џџџџџџџџџџџџџџџџџџ
-__inference_simple_rnn_6_layer_call_fn_241592d!#"?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ


 
p 

 
Њ "џџџџџџџџџ
-__inference_simple_rnn_6_layer_call_fn_241603d!#"?Ђ<
5Ђ2
$!
inputsџџџџџџџџџ


 
p

 
Њ "џџџџџџџџџ
M__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_242080З \ЂY
RЂO
 
inputsџџџџџџџџџ
'Ђ$
"
states/0џџџџџџџџџ

p 
Њ "RЂO
HЂE

0/0џџџџџџџџџ

$!

0/1/0џџџџџџџџџ

 
M__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_242097З \ЂY
RЂO
 
inputsџџџџџџџџџ
'Ђ$
"
states/0џџџџџџџџџ

p
Њ "RЂO
HЂE

0/0џџџџџџџџџ

$!

0/1/0џџџџџџџџџ

 р
2__inference_simple_rnn_cell_5_layer_call_fn_242049Љ \ЂY
RЂO
 
inputsџџџџџџџџџ
'Ђ$
"
states/0џџџџџџџџџ

p 
Њ "DЂA

0џџџџџџџџџ

"

1/0џџџџџџџџџ
р
2__inference_simple_rnn_cell_5_layer_call_fn_242063Љ \ЂY
RЂO
 
inputsџџџџџџџџџ
'Ђ$
"
states/0џџџџџџџџџ

p
Њ "DЂA

0џџџџџџџџџ

"

1/0џџџџџџџџџ

M__inference_simple_rnn_cell_6_layer_call_and_return_conditional_losses_242142З!#"\ЂY
RЂO
 
inputsџџџџџџџџџ

'Ђ$
"
states/0џџџџџџџџџ
p 
Њ "RЂO
HЂE

0/0џџџџџџџџџ
$!

0/1/0џџџџџџџџџ
 
M__inference_simple_rnn_cell_6_layer_call_and_return_conditional_losses_242159З!#"\ЂY
RЂO
 
inputsџџџџџџџџџ

'Ђ$
"
states/0џџџџџџџџџ
p
Њ "RЂO
HЂE

0/0џџџџџџџџџ
$!

0/1/0џџџџџџџџџ
 р
2__inference_simple_rnn_cell_6_layer_call_fn_242111Љ!#"\ЂY
RЂO
 
inputsџџџџџџџџџ

'Ђ$
"
states/0џџџџџџџџџ
p 
Њ "DЂA

0џџџџџџџџџ
"

1/0џџџџџџџџџр
2__inference_simple_rnn_cell_6_layer_call_fn_242125Љ!#"\ЂY
RЂO
 
inputsџџџџџџџџџ

'Ђ$
"
states/0џџџџџџџџџ
p
Њ "DЂA

0џџџџџџџџџ
"

1/0џџџџџџџџџ