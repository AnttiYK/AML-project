»
®ý
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

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
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
Á
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
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.11.02v2.11.0-rc2-17-gd5b57ca93e58ú¥
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
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0
b
total_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
b
count_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_4
[
count_4/Read/ReadVariableOpReadVariableOpcount_4*
_output_shapes
: *
dtype0
b
total_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_4
[
total_4/Read/ReadVariableOpReadVariableOptotal_4*
_output_shapes
: *
dtype0
~
Adam/v/fm_head/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/v/fm_head/bias
w
'Adam/v/fm_head/bias/Read/ReadVariableOpReadVariableOpAdam/v/fm_head/bias*
_output_shapes
:*
dtype0
~
Adam/m/fm_head/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/m/fm_head/bias
w
'Adam/m/fm_head/bias/Read/ReadVariableOpReadVariableOpAdam/m/fm_head/bias*
_output_shapes
:*
dtype0

Adam/v/fm_head/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/v/fm_head/kernel

)Adam/v/fm_head/kernel/Read/ReadVariableOpReadVariableOpAdam/v/fm_head/kernel*
_output_shapes

: *
dtype0

Adam/m/fm_head/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/m/fm_head/kernel

)Adam/m/fm_head/kernel/Read/ReadVariableOpReadVariableOpAdam/m/fm_head/kernel*
_output_shapes

: *
dtype0
~
Adam/v/sm_head/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/v/sm_head/bias
w
'Adam/v/sm_head/bias/Read/ReadVariableOpReadVariableOpAdam/v/sm_head/bias*
_output_shapes
:*
dtype0
~
Adam/m/sm_head/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/m/sm_head/bias
w
'Adam/m/sm_head/bias/Read/ReadVariableOpReadVariableOpAdam/m/sm_head/bias*
_output_shapes
:*
dtype0

Adam/v/sm_head/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_nameAdam/v/sm_head/kernel

)Adam/v/sm_head/kernel/Read/ReadVariableOpReadVariableOpAdam/v/sm_head/kernel*
_output_shapes
:	*
dtype0

Adam/m/sm_head/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_nameAdam/m/sm_head/kernel

)Adam/m/sm_head/kernel/Read/ReadVariableOpReadVariableOpAdam/m/sm_head/kernel*
_output_shapes
:	*
dtype0
v
Adam/v/fm3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameAdam/v/fm3/bias
o
#Adam/v/fm3/bias/Read/ReadVariableOpReadVariableOpAdam/v/fm3/bias*
_output_shapes
: *
dtype0
v
Adam/m/fm3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameAdam/m/fm3/bias
o
#Adam/m/fm3/bias/Read/ReadVariableOpReadVariableOpAdam/m/fm3/bias*
_output_shapes
: *
dtype0
~
Adam/v/fm3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *"
shared_nameAdam/v/fm3/kernel
w
%Adam/v/fm3/kernel/Read/ReadVariableOpReadVariableOpAdam/v/fm3/kernel*
_output_shapes

:@ *
dtype0
~
Adam/m/fm3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *"
shared_nameAdam/m/fm3/kernel
w
%Adam/m/fm3/kernel/Read/ReadVariableOpReadVariableOpAdam/m/fm3/kernel*
_output_shapes

:@ *
dtype0
w
Adam/v/sm1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/v/sm1/bias
p
#Adam/v/sm1/bias/Read/ReadVariableOpReadVariableOpAdam/v/sm1/bias*
_output_shapes	
:*
dtype0
w
Adam/m/sm1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/m/sm1/bias
p
#Adam/m/sm1/bias/Read/ReadVariableOpReadVariableOpAdam/m/sm1/bias*
_output_shapes	
:*
dtype0

Adam/v/sm1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ò*"
shared_nameAdam/v/sm1/kernel
z
%Adam/v/sm1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/sm1/kernel*!
_output_shapes
:ò*
dtype0

Adam/m/sm1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ò*"
shared_nameAdam/m/sm1/kernel
z
%Adam/m/sm1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/sm1/kernel*!
_output_shapes
:ò*
dtype0
v
Adam/v/fm2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameAdam/v/fm2/bias
o
#Adam/v/fm2/bias/Read/ReadVariableOpReadVariableOpAdam/v/fm2/bias*
_output_shapes
:@*
dtype0
v
Adam/m/fm2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameAdam/m/fm2/bias
o
#Adam/m/fm2/bias/Read/ReadVariableOpReadVariableOpAdam/m/fm2/bias*
_output_shapes
:@*
dtype0

Adam/v/fm2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*"
shared_nameAdam/v/fm2/kernel
x
%Adam/v/fm2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/fm2/kernel*
_output_shapes
:	@*
dtype0

Adam/m/fm2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*"
shared_nameAdam/m/fm2/kernel
x
%Adam/m/fm2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/fm2/kernel*
_output_shapes
:	@*
dtype0
w
Adam/v/fm1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/v/fm1/bias
p
#Adam/v/fm1/bias/Read/ReadVariableOpReadVariableOpAdam/v/fm1/bias*
_output_shapes	
:*
dtype0
w
Adam/m/fm1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/m/fm1/bias
p
#Adam/m/fm1/bias/Read/ReadVariableOpReadVariableOpAdam/m/fm1/bias*
_output_shapes	
:*
dtype0

Adam/v/fm1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ò*"
shared_nameAdam/v/fm1/kernel
z
%Adam/v/fm1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/fm1/kernel*!
_output_shapes
:ò*
dtype0

Adam/m/fm1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ò*"
shared_nameAdam/m/fm1/kernel
z
%Adam/m/fm1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/fm1/kernel*!
_output_shapes
:ò*
dtype0
v
Adam/v/bm6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameAdam/v/bm6/bias
o
#Adam/v/bm6/bias/Read/ReadVariableOpReadVariableOpAdam/v/bm6/bias*
_output_shapes
:@*
dtype0
v
Adam/m/bm6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameAdam/m/bm6/bias
o
#Adam/m/bm6/bias/Read/ReadVariableOpReadVariableOpAdam/m/bm6/bias*
_output_shapes
:@*
dtype0

Adam/v/bm6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*"
shared_nameAdam/v/bm6/kernel

%Adam/v/bm6/kernel/Read/ReadVariableOpReadVariableOpAdam/v/bm6/kernel*&
_output_shapes
: @*
dtype0

Adam/m/bm6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*"
shared_nameAdam/m/bm6/kernel

%Adam/m/bm6/kernel/Read/ReadVariableOpReadVariableOpAdam/m/bm6/kernel*&
_output_shapes
: @*
dtype0
v
Adam/v/bm4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameAdam/v/bm4/bias
o
#Adam/v/bm4/bias/Read/ReadVariableOpReadVariableOpAdam/v/bm4/bias*
_output_shapes
: *
dtype0
v
Adam/m/bm4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameAdam/m/bm4/bias
o
#Adam/m/bm4/bias/Read/ReadVariableOpReadVariableOpAdam/m/bm4/bias*
_output_shapes
: *
dtype0

Adam/v/bm4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/v/bm4/kernel

%Adam/v/bm4/kernel/Read/ReadVariableOpReadVariableOpAdam/v/bm4/kernel*&
_output_shapes
: *
dtype0

Adam/m/bm4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/m/bm4/kernel

%Adam/m/bm4/kernel/Read/ReadVariableOpReadVariableOpAdam/m/bm4/kernel*&
_output_shapes
: *
dtype0
v
Adam/v/bm2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/v/bm2/bias
o
#Adam/v/bm2/bias/Read/ReadVariableOpReadVariableOpAdam/v/bm2/bias*
_output_shapes
:*
dtype0
v
Adam/m/bm2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/m/bm2/bias
o
#Adam/m/bm2/bias/Read/ReadVariableOpReadVariableOpAdam/m/bm2/bias*
_output_shapes
:*
dtype0

Adam/v/bm2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/v/bm2/kernel

%Adam/v/bm2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/bm2/kernel*&
_output_shapes
:*
dtype0

Adam/m/bm2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/m/bm2/kernel

%Adam/m/bm2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/bm2/kernel*&
_output_shapes
:*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
p
fm_head/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namefm_head/bias
i
 fm_head/bias/Read/ReadVariableOpReadVariableOpfm_head/bias*
_output_shapes
:*
dtype0
x
fm_head/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namefm_head/kernel
q
"fm_head/kernel/Read/ReadVariableOpReadVariableOpfm_head/kernel*
_output_shapes

: *
dtype0
p
sm_head/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namesm_head/bias
i
 sm_head/bias/Read/ReadVariableOpReadVariableOpsm_head/bias*
_output_shapes
:*
dtype0
y
sm_head/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namesm_head/kernel
r
"sm_head/kernel/Read/ReadVariableOpReadVariableOpsm_head/kernel*
_output_shapes
:	*
dtype0
h
fm3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
fm3/bias
a
fm3/bias/Read/ReadVariableOpReadVariableOpfm3/bias*
_output_shapes
: *
dtype0
p

fm3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *
shared_name
fm3/kernel
i
fm3/kernel/Read/ReadVariableOpReadVariableOp
fm3/kernel*
_output_shapes

:@ *
dtype0
i
sm1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
sm1/bias
b
sm1/bias/Read/ReadVariableOpReadVariableOpsm1/bias*
_output_shapes	
:*
dtype0
s

sm1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ò*
shared_name
sm1/kernel
l
sm1/kernel/Read/ReadVariableOpReadVariableOp
sm1/kernel*!
_output_shapes
:ò*
dtype0
h
fm2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
fm2/bias
a
fm2/bias/Read/ReadVariableOpReadVariableOpfm2/bias*
_output_shapes
:@*
dtype0
q

fm2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*
shared_name
fm2/kernel
j
fm2/kernel/Read/ReadVariableOpReadVariableOp
fm2/kernel*
_output_shapes
:	@*
dtype0
i
fm1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
fm1/bias
b
fm1/bias/Read/ReadVariableOpReadVariableOpfm1/bias*
_output_shapes	
:*
dtype0
s

fm1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ò*
shared_name
fm1/kernel
l
fm1/kernel/Read/ReadVariableOpReadVariableOp
fm1/kernel*!
_output_shapes
:ò*
dtype0
h
bm6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
bm6/bias
a
bm6/bias/Read/ReadVariableOpReadVariableOpbm6/bias*
_output_shapes
:@*
dtype0
x

bm6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*
shared_name
bm6/kernel
q
bm6/kernel/Read/ReadVariableOpReadVariableOp
bm6/kernel*&
_output_shapes
: @*
dtype0
h
bm4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
bm4/bias
a
bm4/bias/Read/ReadVariableOpReadVariableOpbm4/bias*
_output_shapes
: *
dtype0
x

bm4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
bm4/kernel
q
bm4/kernel/Read/ReadVariableOpReadVariableOp
bm4/kernel*&
_output_shapes
: *
dtype0
h
bm2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
bm2/bias
a
bm2/bias/Read/ReadVariableOpReadVariableOpbm2/bias*
_output_shapes
:*
dtype0
x

bm2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
bm2/kernel
q
bm2/kernel/Read/ReadVariableOpReadVariableOp
bm2/kernel*&
_output_shapes
:*
dtype0

serving_default_input_1Placeholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿ´´
¿
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1
bm2/kernelbm2/bias
bm4/kernelbm4/bias
bm6/kernelbm6/bias
fm1/kernelfm1/bias
fm2/kernelfm2/bias
fm3/kernelfm3/bias
sm1/kernelsm1/biasfm_head/kernelfm_head/biassm_head/kernelsm_head/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_87305

NoOpNoOp
È
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value÷Bó Bë

layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer-8

layer_with_weights-3

layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer_with_weights-6
layer-12
layer_with_weights-7
layer-13
layer_with_weights-8
layer-14
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
loss

signatures*
* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
È
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias
 (_jit_compiled_convolution_op*

)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses* 
È
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses

5kernel
6bias
 7_jit_compiled_convolution_op*

8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses* 
È
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses

Dkernel
Ebias
 F_jit_compiled_convolution_op*

G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses* 

M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses* 
¦
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses

Ykernel
Zbias*
¦
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses

akernel
bbias*
¦
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses

ikernel
jbias*
¦
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses

qkernel
rbias*
¦
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses

ykernel
zbias*
©
{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias*

&0
'1
52
63
D4
E5
Y6
Z7
a8
b9
i10
j11
q12
r13
y14
z15
16
17*

&0
'1
52
63
D4
E5
Y6
Z7
a8
b9
i10
j11
q12
r13
y14
z15
16
17*
* 
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
trace_0
trace_1
trace_2
trace_3* 
:
trace_0
trace_1
trace_2
trace_3* 
* 


_variables
_iterations
_learning_rate
_index_dict

_momentums
_velocities
_update_step_xla*
* 

serving_default* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

trace_0* 

trace_0* 

&0
'1*

&0
'1*
* 

non_trainable_variables
 layers
¡metrics
 ¢layer_regularization_losses
£layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*

¤trace_0* 

¥trace_0* 
ZT
VARIABLE_VALUE
bm2/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEbm2/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

¦non_trainable_variables
§layers
¨metrics
 ©layer_regularization_losses
ªlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses* 

«trace_0* 

¬trace_0* 

50
61*

50
61*
* 

­non_trainable_variables
®layers
¯metrics
 °layer_regularization_losses
±layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses*

²trace_0* 

³trace_0* 
ZT
VARIABLE_VALUE
bm4/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEbm4/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

´non_trainable_variables
µlayers
¶metrics
 ·layer_regularization_losses
¸layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses* 

¹trace_0* 

ºtrace_0* 

D0
E1*

D0
E1*
* 

»non_trainable_variables
¼layers
½metrics
 ¾layer_regularization_losses
¿layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses*

Àtrace_0* 

Átrace_0* 
ZT
VARIABLE_VALUE
bm6/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEbm6/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

Ânon_trainable_variables
Ãlayers
Ämetrics
 Ålayer_regularization_losses
Ælayer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses* 

Çtrace_0* 

Ètrace_0* 
* 
* 
* 

Énon_trainable_variables
Êlayers
Ëmetrics
 Ìlayer_regularization_losses
Ílayer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses* 

Îtrace_0* 

Ïtrace_0* 

Y0
Z1*

Y0
Z1*
* 

Ðnon_trainable_variables
Ñlayers
Òmetrics
 Ólayer_regularization_losses
Ôlayer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses*

Õtrace_0* 

Ötrace_0* 
ZT
VARIABLE_VALUE
fm1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEfm1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

a0
b1*

a0
b1*
* 

×non_trainable_variables
Ølayers
Ùmetrics
 Úlayer_regularization_losses
Ûlayer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses*

Ütrace_0* 

Ýtrace_0* 
ZT
VARIABLE_VALUE
fm2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEfm2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

i0
j1*

i0
j1*
* 

Þnon_trainable_variables
ßlayers
àmetrics
 álayer_regularization_losses
âlayer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses*

ãtrace_0* 

ätrace_0* 
ZT
VARIABLE_VALUE
sm1/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEsm1/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

q0
r1*

q0
r1*
* 

ånon_trainable_variables
ælayers
çmetrics
 èlayer_regularization_losses
élayer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses*

êtrace_0* 

ëtrace_0* 
ZT
VARIABLE_VALUE
fm3/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEfm3/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

y0
z1*

y0
z1*
* 

ìnon_trainable_variables
ílayers
îmetrics
 ïlayer_regularization_losses
ðlayer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses*

ñtrace_0* 

òtrace_0* 
^X
VARIABLE_VALUEsm_head/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEsm_head/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

ónon_trainable_variables
ôlayers
õmetrics
 ölayer_regularization_losses
÷layer_metrics
{	variables
|trainable_variables
}regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

øtrace_0* 

ùtrace_0* 
^X
VARIABLE_VALUEfm_head/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEfm_head/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
r
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14*
,
ú0
û1
ü2
ý3
þ4*
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
Ç
0
ÿ1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
 34
¡35
¢36*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 

ÿ0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
¡17*

0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
 16
¢17*
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
* 
* 
* 
* 
<
£	variables
¤	keras_api

¥total

¦count*
<
§	variables
¨	keras_api

©total

ªcount*
<
«	variables
¬	keras_api

­total

®count*
M
¯	variables
°	keras_api

±total

²count
³
_fn_kwargs*
M
´	variables
µ	keras_api

¶total

·count
¸
_fn_kwargs*
\V
VARIABLE_VALUEAdam/m/bm2/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/v/bm2/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEAdam/m/bm2/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEAdam/v/bm2/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/m/bm4/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/v/bm4/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEAdam/m/bm4/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEAdam/v/bm4/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEAdam/m/bm6/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/bm6/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/m/bm6/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/v/bm6/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/fm1/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/fm1/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/m/fm1/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/v/fm1/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/fm2/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/fm2/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/m/fm2/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/v/fm2/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/sm1/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/sm1/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/m/sm1/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/v/sm1/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/fm3/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/fm3/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/m/fm3/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/v/fm3/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/sm_head/kernel2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/sm_head/kernel2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/sm_head/bias2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/sm_head/bias2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/fm_head/kernel2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/fm_head/kernel2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/fm_head/bias2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/fm_head/bias2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*

¥0
¦1*

£	variables*
UO
VARIABLE_VALUEtotal_44keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_44keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

©0
ª1*

§	variables*
UO
VARIABLE_VALUEtotal_34keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_34keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*

­0
®1*

«	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*

±0
²1*

¯	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

¶0
·1*

´	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ò
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamebm2/kernel/Read/ReadVariableOpbm2/bias/Read/ReadVariableOpbm4/kernel/Read/ReadVariableOpbm4/bias/Read/ReadVariableOpbm6/kernel/Read/ReadVariableOpbm6/bias/Read/ReadVariableOpfm1/kernel/Read/ReadVariableOpfm1/bias/Read/ReadVariableOpfm2/kernel/Read/ReadVariableOpfm2/bias/Read/ReadVariableOpsm1/kernel/Read/ReadVariableOpsm1/bias/Read/ReadVariableOpfm3/kernel/Read/ReadVariableOpfm3/bias/Read/ReadVariableOp"sm_head/kernel/Read/ReadVariableOp sm_head/bias/Read/ReadVariableOp"fm_head/kernel/Read/ReadVariableOp fm_head/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp%Adam/m/bm2/kernel/Read/ReadVariableOp%Adam/v/bm2/kernel/Read/ReadVariableOp#Adam/m/bm2/bias/Read/ReadVariableOp#Adam/v/bm2/bias/Read/ReadVariableOp%Adam/m/bm4/kernel/Read/ReadVariableOp%Adam/v/bm4/kernel/Read/ReadVariableOp#Adam/m/bm4/bias/Read/ReadVariableOp#Adam/v/bm4/bias/Read/ReadVariableOp%Adam/m/bm6/kernel/Read/ReadVariableOp%Adam/v/bm6/kernel/Read/ReadVariableOp#Adam/m/bm6/bias/Read/ReadVariableOp#Adam/v/bm6/bias/Read/ReadVariableOp%Adam/m/fm1/kernel/Read/ReadVariableOp%Adam/v/fm1/kernel/Read/ReadVariableOp#Adam/m/fm1/bias/Read/ReadVariableOp#Adam/v/fm1/bias/Read/ReadVariableOp%Adam/m/fm2/kernel/Read/ReadVariableOp%Adam/v/fm2/kernel/Read/ReadVariableOp#Adam/m/fm2/bias/Read/ReadVariableOp#Adam/v/fm2/bias/Read/ReadVariableOp%Adam/m/sm1/kernel/Read/ReadVariableOp%Adam/v/sm1/kernel/Read/ReadVariableOp#Adam/m/sm1/bias/Read/ReadVariableOp#Adam/v/sm1/bias/Read/ReadVariableOp%Adam/m/fm3/kernel/Read/ReadVariableOp%Adam/v/fm3/kernel/Read/ReadVariableOp#Adam/m/fm3/bias/Read/ReadVariableOp#Adam/v/fm3/bias/Read/ReadVariableOp)Adam/m/sm_head/kernel/Read/ReadVariableOp)Adam/v/sm_head/kernel/Read/ReadVariableOp'Adam/m/sm_head/bias/Read/ReadVariableOp'Adam/v/sm_head/bias/Read/ReadVariableOp)Adam/m/fm_head/kernel/Read/ReadVariableOp)Adam/v/fm_head/kernel/Read/ReadVariableOp'Adam/m/fm_head/bias/Read/ReadVariableOp'Adam/v/fm_head/bias/Read/ReadVariableOptotal_4/Read/ReadVariableOpcount_4/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_3/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*O
TinH
F2D	*
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
GPU 2J 8 *'
f"R 
__inference__traced_save_87998
Å

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename
bm2/kernelbm2/bias
bm4/kernelbm4/bias
bm6/kernelbm6/bias
fm1/kernelfm1/bias
fm2/kernelfm2/bias
sm1/kernelsm1/bias
fm3/kernelfm3/biassm_head/kernelsm_head/biasfm_head/kernelfm_head/bias	iterationlearning_rateAdam/m/bm2/kernelAdam/v/bm2/kernelAdam/m/bm2/biasAdam/v/bm2/biasAdam/m/bm4/kernelAdam/v/bm4/kernelAdam/m/bm4/biasAdam/v/bm4/biasAdam/m/bm6/kernelAdam/v/bm6/kernelAdam/m/bm6/biasAdam/v/bm6/biasAdam/m/fm1/kernelAdam/v/fm1/kernelAdam/m/fm1/biasAdam/v/fm1/biasAdam/m/fm2/kernelAdam/v/fm2/kernelAdam/m/fm2/biasAdam/v/fm2/biasAdam/m/sm1/kernelAdam/v/sm1/kernelAdam/m/sm1/biasAdam/v/sm1/biasAdam/m/fm3/kernelAdam/v/fm3/kernelAdam/m/fm3/biasAdam/v/fm3/biasAdam/m/sm_head/kernelAdam/v/sm_head/kernelAdam/m/sm_head/biasAdam/v/sm_head/biasAdam/m/fm_head/kernelAdam/v/fm_head/kernelAdam/m/fm_head/biasAdam/v/fm_head/biastotal_4count_4total_3count_3total_2count_2total_1count_1totalcount*N
TinG
E2C*
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
GPU 2J 8 **
f%R#
!__inference__traced_restore_88206


?
#__inference_bm5_layer_call_fn_87611

inputs
identityÌ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_bm5_layer_call_and_return_conditional_losses_86624
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¹
ú
#__inference_signature_wrapper_87305
input_1!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@
	unknown_5:ò
	unknown_6:	
	unknown_7:	@
	unknown_8:@
	unknown_9:@ 

unknown_10: 

unknown_11:ò

unknown_12:	

unknown_13: 

unknown_14:

unknown_15:	

unknown_16:
identity

identity_1¢StatefulPartitionedCall¡
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_86603o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ´´: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
!
_user_specified_name	input_1
Û
ü
%__inference_model_layer_call_fn_87148
input_1!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@
	unknown_5:ò
	unknown_6:	
	unknown_7:	@
	unknown_8:@
	unknown_9:@ 

unknown_10: 

unknown_11:ò

unknown_12:	

unknown_13: 

unknown_14:

unknown_15:	

unknown_16:
identity

identity_1¢StatefulPartitionedCallÁ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_87064o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ´´: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
!
_user_specified_name	input_1
À

#__inference_sm1_layer_call_fn_87706

inputs
unknown:ò
	unknown_0:	
identity¢StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_sm1_layer_call_and_return_conditional_losses_86780p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿò: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
 
_user_specified_nameinputs
¡:

@__inference_model_layer_call_and_return_conditional_losses_87203
input_1#
	bm2_87152:
	bm2_87154:#
	bm4_87158: 
	bm4_87160: #
	bm6_87164: @
	bm6_87166:@
	fm1_87171:ò
	fm1_87173:	
	fm2_87176:	@
	fm2_87178:@
	fm3_87181:@ 
	fm3_87183: 
	sm1_87186:ò
	sm1_87188:	
fm_head_87191: 
fm_head_87193: 
sm_head_87196:	
sm_head_87198:
identity

identity_1¢bm2/StatefulPartitionedCall¢bm4/StatefulPartitionedCall¢bm6/StatefulPartitionedCall¢fm1/StatefulPartitionedCall¢fm2/StatefulPartitionedCall¢fm3/StatefulPartitionedCall¢fm_head/StatefulPartitionedCall¢sm1/StatefulPartitionedCall¢sm_head/StatefulPartitionedCall¸
bm1/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_bm1_layer_call_and_return_conditional_losses_86654ù
bm2/StatefulPartitionedCallStatefulPartitionedCallbm1/PartitionedCall:output:0	bm2_87152	bm2_87154*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_bm2_layer_call_and_return_conditional_losses_86667Ó
bm3/PartitionedCallPartitionedCall$bm2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿZZ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_bm3_layer_call_and_return_conditional_losses_86612÷
bm4/StatefulPartitionedCallStatefulPartitionedCallbm3/PartitionedCall:output:0	bm4_87158	bm4_87160*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿZZ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_bm4_layer_call_and_return_conditional_losses_86685Ó
bm5/PartitionedCallPartitionedCall$bm4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-- * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_bm5_layer_call_and_return_conditional_losses_86624÷
bm6/StatefulPartitionedCallStatefulPartitionedCallbm5/PartitionedCall:output:0	bm6_87164	bm6_87166*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_bm6_layer_call_and_return_conditional_losses_86703Ó
bm7/PartitionedCallPartitionedCall$bm6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_bm7_layer_call_and_return_conditional_losses_86636Å
bm8/PartitionedCallPartitionedCallbm7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_bm8_layer_call_and_return_conditional_losses_86716ð
fm1/StatefulPartitionedCallStatefulPartitionedCallbm8/PartitionedCall:output:0	fm1_87171	fm1_87173*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_fm1_layer_call_and_return_conditional_losses_86729÷
fm2/StatefulPartitionedCallStatefulPartitionedCall$fm1/StatefulPartitionedCall:output:0	fm2_87176	fm2_87178*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_fm2_layer_call_and_return_conditional_losses_86746÷
fm3/StatefulPartitionedCallStatefulPartitionedCall$fm2/StatefulPartitionedCall:output:0	fm3_87181	fm3_87183*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_fm3_layer_call_and_return_conditional_losses_86763ð
sm1/StatefulPartitionedCallStatefulPartitionedCallbm8/PartitionedCall:output:0	sm1_87186	sm1_87188*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_sm1_layer_call_and_return_conditional_losses_86780
fm_head/StatefulPartitionedCallStatefulPartitionedCall$fm3/StatefulPartitionedCall:output:0fm_head_87191fm_head_87193*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_fm_head_layer_call_and_return_conditional_losses_86797
sm_head/StatefulPartitionedCallStatefulPartitionedCall$sm1/StatefulPartitionedCall:output:0sm_head_87196sm_head_87198*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_sm_head_layer_call_and_return_conditional_losses_86813w
IdentityIdentity(sm_head/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy

Identity_1Identity(fm_head/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
NoOpNoOp^bm2/StatefulPartitionedCall^bm4/StatefulPartitionedCall^bm6/StatefulPartitionedCall^fm1/StatefulPartitionedCall^fm2/StatefulPartitionedCall^fm3/StatefulPartitionedCall ^fm_head/StatefulPartitionedCall^sm1/StatefulPartitionedCall ^sm_head/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ´´: : : : : : : : : : : : : : : : : : 2:
bm2/StatefulPartitionedCallbm2/StatefulPartitionedCall2:
bm4/StatefulPartitionedCallbm4/StatefulPartitionedCall2:
bm6/StatefulPartitionedCallbm6/StatefulPartitionedCall2:
fm1/StatefulPartitionedCallfm1/StatefulPartitionedCall2:
fm2/StatefulPartitionedCallfm2/StatefulPartitionedCall2:
fm3/StatefulPartitionedCallfm3/StatefulPartitionedCall2B
fm_head/StatefulPartitionedCallfm_head/StatefulPartitionedCall2:
sm1/StatefulPartitionedCallsm1/StatefulPartitionedCall2B
sm_head/StatefulPartitionedCallsm_head/StatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
!
_user_specified_name	input_1


ð
>__inference_fm2_layer_call_and_return_conditional_losses_87697

inputs1
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

?
#__inference_bm7_layer_call_fn_87641

inputs
identityÌ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_bm7_layer_call_and_return_conditional_losses_86636
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Þ

#__inference_bm6_layer_call_fn_87625

inputs!
unknown: @
	unknown_0:@
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_bm6_layer_call_and_return_conditional_losses_86703w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ-- : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-- 
 
_user_specified_nameinputs
É	
ô
B__inference_sm_head_layer_call_and_return_conditional_losses_87756

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¶

#__inference_fm3_layer_call_fn_87726

inputs
unknown:@ 
	unknown_0: 
identity¢StatefulPartitionedCallÓ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_fm3_layer_call_and_return_conditional_losses_86763o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ð
Z
>__inference_bm1_layer_call_and_return_conditional_losses_87556

inputs
identityK
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    _
mulMulinputsCast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´d
addAddV2mul:z:0Cast_1/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´Y
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´´:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs
ð
Z
>__inference_bm1_layer_call_and_return_conditional_losses_86654

inputs
identityK
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    _
mulMulinputsCast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´d
addAddV2mul:z:0Cast_1/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´Y
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´´:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs
É	
ô
B__inference_sm_head_layer_call_and_return_conditional_losses_86813

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

?
#__inference_bm3_layer_call_fn_87581

inputs
identityÌ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_bm3_layer_call_and_return_conditional_losses_86612
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Z
>__inference_bm5_layer_call_and_return_conditional_losses_87616

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
:

@__inference_model_layer_call_and_return_conditional_losses_87064

inputs#
	bm2_87013:
	bm2_87015:#
	bm4_87019: 
	bm4_87021: #
	bm6_87025: @
	bm6_87027:@
	fm1_87032:ò
	fm1_87034:	
	fm2_87037:	@
	fm2_87039:@
	fm3_87042:@ 
	fm3_87044: 
	sm1_87047:ò
	sm1_87049:	
fm_head_87052: 
fm_head_87054: 
sm_head_87057:	
sm_head_87059:
identity

identity_1¢bm2/StatefulPartitionedCall¢bm4/StatefulPartitionedCall¢bm6/StatefulPartitionedCall¢fm1/StatefulPartitionedCall¢fm2/StatefulPartitionedCall¢fm3/StatefulPartitionedCall¢fm_head/StatefulPartitionedCall¢sm1/StatefulPartitionedCall¢sm_head/StatefulPartitionedCall·
bm1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_bm1_layer_call_and_return_conditional_losses_86654ù
bm2/StatefulPartitionedCallStatefulPartitionedCallbm1/PartitionedCall:output:0	bm2_87013	bm2_87015*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_bm2_layer_call_and_return_conditional_losses_86667Ó
bm3/PartitionedCallPartitionedCall$bm2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿZZ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_bm3_layer_call_and_return_conditional_losses_86612÷
bm4/StatefulPartitionedCallStatefulPartitionedCallbm3/PartitionedCall:output:0	bm4_87019	bm4_87021*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿZZ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_bm4_layer_call_and_return_conditional_losses_86685Ó
bm5/PartitionedCallPartitionedCall$bm4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-- * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_bm5_layer_call_and_return_conditional_losses_86624÷
bm6/StatefulPartitionedCallStatefulPartitionedCallbm5/PartitionedCall:output:0	bm6_87025	bm6_87027*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_bm6_layer_call_and_return_conditional_losses_86703Ó
bm7/PartitionedCallPartitionedCall$bm6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_bm7_layer_call_and_return_conditional_losses_86636Å
bm8/PartitionedCallPartitionedCallbm7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_bm8_layer_call_and_return_conditional_losses_86716ð
fm1/StatefulPartitionedCallStatefulPartitionedCallbm8/PartitionedCall:output:0	fm1_87032	fm1_87034*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_fm1_layer_call_and_return_conditional_losses_86729÷
fm2/StatefulPartitionedCallStatefulPartitionedCall$fm1/StatefulPartitionedCall:output:0	fm2_87037	fm2_87039*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_fm2_layer_call_and_return_conditional_losses_86746÷
fm3/StatefulPartitionedCallStatefulPartitionedCall$fm2/StatefulPartitionedCall:output:0	fm3_87042	fm3_87044*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_fm3_layer_call_and_return_conditional_losses_86763ð
sm1/StatefulPartitionedCallStatefulPartitionedCallbm8/PartitionedCall:output:0	sm1_87047	sm1_87049*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_sm1_layer_call_and_return_conditional_losses_86780
fm_head/StatefulPartitionedCallStatefulPartitionedCall$fm3/StatefulPartitionedCall:output:0fm_head_87052fm_head_87054*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_fm_head_layer_call_and_return_conditional_losses_86797
sm_head/StatefulPartitionedCallStatefulPartitionedCall$sm1/StatefulPartitionedCall:output:0sm_head_87057sm_head_87059*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_sm_head_layer_call_and_return_conditional_losses_86813w
IdentityIdentity(sm_head/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy

Identity_1Identity(fm_head/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
NoOpNoOp^bm2/StatefulPartitionedCall^bm4/StatefulPartitionedCall^bm6/StatefulPartitionedCall^fm1/StatefulPartitionedCall^fm2/StatefulPartitionedCall^fm3/StatefulPartitionedCall ^fm_head/StatefulPartitionedCall^sm1/StatefulPartitionedCall ^sm_head/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ´´: : : : : : : : : : : : : : : : : : 2:
bm2/StatefulPartitionedCallbm2/StatefulPartitionedCall2:
bm4/StatefulPartitionedCallbm4/StatefulPartitionedCall2:
bm6/StatefulPartitionedCallbm6/StatefulPartitionedCall2:
fm1/StatefulPartitionedCallfm1/StatefulPartitionedCall2:
fm2/StatefulPartitionedCallfm2/StatefulPartitionedCall2:
fm3/StatefulPartitionedCallfm3/StatefulPartitionedCall2B
fm_head/StatefulPartitionedCallfm_head/StatefulPartitionedCall2:
sm1/StatefulPartitionedCallsm1/StatefulPartitionedCall2B
sm_head/StatefulPartitionedCallsm_head/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs
:

@__inference_model_layer_call_and_return_conditional_losses_86821

inputs#
	bm2_86668:
	bm2_86670:#
	bm4_86686: 
	bm4_86688: #
	bm6_86704: @
	bm6_86706:@
	fm1_86730:ò
	fm1_86732:	
	fm2_86747:	@
	fm2_86749:@
	fm3_86764:@ 
	fm3_86766: 
	sm1_86781:ò
	sm1_86783:	
fm_head_86798: 
fm_head_86800: 
sm_head_86814:	
sm_head_86816:
identity

identity_1¢bm2/StatefulPartitionedCall¢bm4/StatefulPartitionedCall¢bm6/StatefulPartitionedCall¢fm1/StatefulPartitionedCall¢fm2/StatefulPartitionedCall¢fm3/StatefulPartitionedCall¢fm_head/StatefulPartitionedCall¢sm1/StatefulPartitionedCall¢sm_head/StatefulPartitionedCall·
bm1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_bm1_layer_call_and_return_conditional_losses_86654ù
bm2/StatefulPartitionedCallStatefulPartitionedCallbm1/PartitionedCall:output:0	bm2_86668	bm2_86670*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_bm2_layer_call_and_return_conditional_losses_86667Ó
bm3/PartitionedCallPartitionedCall$bm2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿZZ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_bm3_layer_call_and_return_conditional_losses_86612÷
bm4/StatefulPartitionedCallStatefulPartitionedCallbm3/PartitionedCall:output:0	bm4_86686	bm4_86688*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿZZ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_bm4_layer_call_and_return_conditional_losses_86685Ó
bm5/PartitionedCallPartitionedCall$bm4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-- * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_bm5_layer_call_and_return_conditional_losses_86624÷
bm6/StatefulPartitionedCallStatefulPartitionedCallbm5/PartitionedCall:output:0	bm6_86704	bm6_86706*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_bm6_layer_call_and_return_conditional_losses_86703Ó
bm7/PartitionedCallPartitionedCall$bm6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_bm7_layer_call_and_return_conditional_losses_86636Å
bm8/PartitionedCallPartitionedCallbm7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_bm8_layer_call_and_return_conditional_losses_86716ð
fm1/StatefulPartitionedCallStatefulPartitionedCallbm8/PartitionedCall:output:0	fm1_86730	fm1_86732*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_fm1_layer_call_and_return_conditional_losses_86729÷
fm2/StatefulPartitionedCallStatefulPartitionedCall$fm1/StatefulPartitionedCall:output:0	fm2_86747	fm2_86749*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_fm2_layer_call_and_return_conditional_losses_86746÷
fm3/StatefulPartitionedCallStatefulPartitionedCall$fm2/StatefulPartitionedCall:output:0	fm3_86764	fm3_86766*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_fm3_layer_call_and_return_conditional_losses_86763ð
sm1/StatefulPartitionedCallStatefulPartitionedCallbm8/PartitionedCall:output:0	sm1_86781	sm1_86783*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_sm1_layer_call_and_return_conditional_losses_86780
fm_head/StatefulPartitionedCallStatefulPartitionedCall$fm3/StatefulPartitionedCall:output:0fm_head_86798fm_head_86800*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_fm_head_layer_call_and_return_conditional_losses_86797
sm_head/StatefulPartitionedCallStatefulPartitionedCall$sm1/StatefulPartitionedCall:output:0sm_head_86814sm_head_86816*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_sm_head_layer_call_and_return_conditional_losses_86813w
IdentityIdentity(sm_head/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy

Identity_1Identity(fm_head/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
NoOpNoOp^bm2/StatefulPartitionedCall^bm4/StatefulPartitionedCall^bm6/StatefulPartitionedCall^fm1/StatefulPartitionedCall^fm2/StatefulPartitionedCall^fm3/StatefulPartitionedCall ^fm_head/StatefulPartitionedCall^sm1/StatefulPartitionedCall ^sm_head/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ´´: : : : : : : : : : : : : : : : : : 2:
bm2/StatefulPartitionedCallbm2/StatefulPartitionedCall2:
bm4/StatefulPartitionedCallbm4/StatefulPartitionedCall2:
bm6/StatefulPartitionedCallbm6/StatefulPartitionedCall2:
fm1/StatefulPartitionedCallfm1/StatefulPartitionedCall2:
fm2/StatefulPartitionedCallfm2/StatefulPartitionedCall2:
fm3/StatefulPartitionedCallfm3/StatefulPartitionedCall2B
fm_head/StatefulPartitionedCallfm_head/StatefulPartitionedCall2:
sm1/StatefulPartitionedCallsm1/StatefulPartitionedCall2B
sm_head/StatefulPartitionedCallsm_head/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs

Z
>__inference_bm7_layer_call_and_return_conditional_losses_87646

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Á

'__inference_sm_head_layer_call_fn_87746

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_sm_head_layer_call_and_return_conditional_losses_86813o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ý

÷
>__inference_bm6_layer_call_and_return_conditional_losses_87636

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ-- : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-- 
 
_user_specified_nameinputs


ó
B__inference_fm_head_layer_call_and_return_conditional_losses_86797

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¡:

@__inference_model_layer_call_and_return_conditional_losses_87258
input_1#
	bm2_87207:
	bm2_87209:#
	bm4_87213: 
	bm4_87215: #
	bm6_87219: @
	bm6_87221:@
	fm1_87226:ò
	fm1_87228:	
	fm2_87231:	@
	fm2_87233:@
	fm3_87236:@ 
	fm3_87238: 
	sm1_87241:ò
	sm1_87243:	
fm_head_87246: 
fm_head_87248: 
sm_head_87251:	
sm_head_87253:
identity

identity_1¢bm2/StatefulPartitionedCall¢bm4/StatefulPartitionedCall¢bm6/StatefulPartitionedCall¢fm1/StatefulPartitionedCall¢fm2/StatefulPartitionedCall¢fm3/StatefulPartitionedCall¢fm_head/StatefulPartitionedCall¢sm1/StatefulPartitionedCall¢sm_head/StatefulPartitionedCall¸
bm1/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_bm1_layer_call_and_return_conditional_losses_86654ù
bm2/StatefulPartitionedCallStatefulPartitionedCallbm1/PartitionedCall:output:0	bm2_87207	bm2_87209*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_bm2_layer_call_and_return_conditional_losses_86667Ó
bm3/PartitionedCallPartitionedCall$bm2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿZZ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_bm3_layer_call_and_return_conditional_losses_86612÷
bm4/StatefulPartitionedCallStatefulPartitionedCallbm3/PartitionedCall:output:0	bm4_87213	bm4_87215*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿZZ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_bm4_layer_call_and_return_conditional_losses_86685Ó
bm5/PartitionedCallPartitionedCall$bm4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-- * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_bm5_layer_call_and_return_conditional_losses_86624÷
bm6/StatefulPartitionedCallStatefulPartitionedCallbm5/PartitionedCall:output:0	bm6_87219	bm6_87221*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_bm6_layer_call_and_return_conditional_losses_86703Ó
bm7/PartitionedCallPartitionedCall$bm6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_bm7_layer_call_and_return_conditional_losses_86636Å
bm8/PartitionedCallPartitionedCallbm7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_bm8_layer_call_and_return_conditional_losses_86716ð
fm1/StatefulPartitionedCallStatefulPartitionedCallbm8/PartitionedCall:output:0	fm1_87226	fm1_87228*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_fm1_layer_call_and_return_conditional_losses_86729÷
fm2/StatefulPartitionedCallStatefulPartitionedCall$fm1/StatefulPartitionedCall:output:0	fm2_87231	fm2_87233*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_fm2_layer_call_and_return_conditional_losses_86746÷
fm3/StatefulPartitionedCallStatefulPartitionedCall$fm2/StatefulPartitionedCall:output:0	fm3_87236	fm3_87238*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_fm3_layer_call_and_return_conditional_losses_86763ð
sm1/StatefulPartitionedCallStatefulPartitionedCallbm8/PartitionedCall:output:0	sm1_87241	sm1_87243*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_sm1_layer_call_and_return_conditional_losses_86780
fm_head/StatefulPartitionedCallStatefulPartitionedCall$fm3/StatefulPartitionedCall:output:0fm_head_87246fm_head_87248*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_fm_head_layer_call_and_return_conditional_losses_86797
sm_head/StatefulPartitionedCallStatefulPartitionedCall$sm1/StatefulPartitionedCall:output:0sm_head_87251sm_head_87253*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_sm_head_layer_call_and_return_conditional_losses_86813w
IdentityIdentity(sm_head/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy

Identity_1Identity(fm_head/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
NoOpNoOp^bm2/StatefulPartitionedCall^bm4/StatefulPartitionedCall^bm6/StatefulPartitionedCall^fm1/StatefulPartitionedCall^fm2/StatefulPartitionedCall^fm3/StatefulPartitionedCall ^fm_head/StatefulPartitionedCall^sm1/StatefulPartitionedCall ^sm_head/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ´´: : : : : : : : : : : : : : : : : : 2:
bm2/StatefulPartitionedCallbm2/StatefulPartitionedCall2:
bm4/StatefulPartitionedCallbm4/StatefulPartitionedCall2:
bm6/StatefulPartitionedCallbm6/StatefulPartitionedCall2:
fm1/StatefulPartitionedCallfm1/StatefulPartitionedCall2:
fm2/StatefulPartitionedCallfm2/StatefulPartitionedCall2:
fm3/StatefulPartitionedCallfm3/StatefulPartitionedCall2B
fm_head/StatefulPartitionedCallfm_head/StatefulPartitionedCall2:
sm1/StatefulPartitionedCallsm1/StatefulPartitionedCall2B
sm_head/StatefulPartitionedCallsm_head/StatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
!
_user_specified_name	input_1


ð
>__inference_fm2_layer_call_and_return_conditional_losses_86746

inputs1
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø
û
%__inference_model_layer_call_fn_87348

inputs!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@
	unknown_5:ò
	unknown_6:	
	unknown_7:	@
	unknown_8:@
	unknown_9:@ 

unknown_10: 

unknown_11:ò

unknown_12:	

unknown_13: 

unknown_14:

unknown_15:	

unknown_16:
identity

identity_1¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_86821o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ´´: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs
¹

#__inference_fm2_layer_call_fn_87686

inputs
unknown:	@
	unknown_0:@
identity¢StatefulPartitionedCallÓ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_fm2_layer_call_and_return_conditional_losses_86746o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø
û
%__inference_model_layer_call_fn_87391

inputs!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@
	unknown_5:ò
	unknown_6:	
	unknown_7:	@
	unknown_8:@
	unknown_9:@ 

unknown_10: 

unknown_11:ò

unknown_12:	

unknown_13: 

unknown_14:

unknown_15:	

unknown_16:
identity

identity_1¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_87064o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ´´: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs

Z
>__inference_bm3_layer_call_and_return_conditional_losses_86612

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
R
ã
@__inference_model_layer_call_and_return_conditional_losses_87543

inputs<
"bm2_conv2d_readvariableop_resource:1
#bm2_biasadd_readvariableop_resource:<
"bm4_conv2d_readvariableop_resource: 1
#bm4_biasadd_readvariableop_resource: <
"bm6_conv2d_readvariableop_resource: @1
#bm6_biasadd_readvariableop_resource:@7
"fm1_matmul_readvariableop_resource:ò2
#fm1_biasadd_readvariableop_resource:	5
"fm2_matmul_readvariableop_resource:	@1
#fm2_biasadd_readvariableop_resource:@4
"fm3_matmul_readvariableop_resource:@ 1
#fm3_biasadd_readvariableop_resource: 7
"sm1_matmul_readvariableop_resource:ò2
#sm1_biasadd_readvariableop_resource:	8
&fm_head_matmul_readvariableop_resource: 5
'fm_head_biasadd_readvariableop_resource:9
&sm_head_matmul_readvariableop_resource:	5
'sm_head_biasadd_readvariableop_resource:
identity

identity_1¢bm2/BiasAdd/ReadVariableOp¢bm2/Conv2D/ReadVariableOp¢bm4/BiasAdd/ReadVariableOp¢bm4/Conv2D/ReadVariableOp¢bm6/BiasAdd/ReadVariableOp¢bm6/Conv2D/ReadVariableOp¢fm1/BiasAdd/ReadVariableOp¢fm1/MatMul/ReadVariableOp¢fm2/BiasAdd/ReadVariableOp¢fm2/MatMul/ReadVariableOp¢fm3/BiasAdd/ReadVariableOp¢fm3/MatMul/ReadVariableOp¢fm_head/BiasAdd/ReadVariableOp¢fm_head/MatMul/ReadVariableOp¢sm1/BiasAdd/ReadVariableOp¢sm1/MatMul/ReadVariableOp¢sm_head/BiasAdd/ReadVariableOp¢sm_head/MatMul/ReadVariableOpO

bm1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;Q
bm1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    g
bm1/mulMulinputsbm1/Cast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´p
bm1/addAddV2bm1/mul:z:0bm1/Cast_1/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
bm2/Conv2D/ReadVariableOpReadVariableOp"bm2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0¨

bm2/Conv2DConv2Dbm1/add:z:0!bm2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´*
paddingSAME*
strides
z
bm2/BiasAdd/ReadVariableOpReadVariableOp#bm2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
bm2/BiasAddBiasAddbm2/Conv2D:output:0"bm2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´b
bm2/ReluRelubm2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
bm3/MaxPoolMaxPoolbm2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿZZ*
ksize
*
paddingVALID*
strides

bm4/Conv2D/ReadVariableOpReadVariableOp"bm4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0¯

bm4/Conv2DConv2Dbm3/MaxPool:output:0!bm4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿZZ *
paddingSAME*
strides
z
bm4/BiasAdd/ReadVariableOpReadVariableOp#bm4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
bm4/BiasAddBiasAddbm4/Conv2D:output:0"bm4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿZZ `
bm4/ReluRelubm4/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿZZ 
bm5/MaxPoolMaxPoolbm4/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-- *
ksize
*
paddingVALID*
strides

bm6/Conv2D/ReadVariableOpReadVariableOp"bm6_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0¯

bm6/Conv2DConv2Dbm5/MaxPool:output:0!bm6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--@*
paddingSAME*
strides
z
bm6/BiasAdd/ReadVariableOpReadVariableOp#bm6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
bm6/BiasAddBiasAddbm6/Conv2D:output:0"bm6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--@`
bm6/ReluRelubm6/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--@
bm7/MaxPoolMaxPoolbm6/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
Z
	bm8/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ y  t
bm8/ReshapeReshapebm7/MaxPool:output:0bm8/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
fm1/MatMul/ReadVariableOpReadVariableOp"fm1_matmul_readvariableop_resource*!
_output_shapes
:ò*
dtype0

fm1/MatMulMatMulbm8/Reshape:output:0!fm1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
fm1/BiasAdd/ReadVariableOpReadVariableOp#fm1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
fm1/BiasAddBiasAddfm1/MatMul:product:0"fm1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
fm1/ReluRelufm1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
fm2/MatMul/ReadVariableOpReadVariableOp"fm2_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0

fm2/MatMulMatMulfm1/Relu:activations:0!fm2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
fm2/BiasAdd/ReadVariableOpReadVariableOp#fm2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
fm2/BiasAddBiasAddfm2/MatMul:product:0"fm2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@X
fm2/ReluRelufm2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
fm3/MatMul/ReadVariableOpReadVariableOp"fm3_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0

fm3/MatMulMatMulfm2/Relu:activations:0!fm3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
fm3/BiasAdd/ReadVariableOpReadVariableOp#fm3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
fm3/BiasAddBiasAddfm3/MatMul:product:0"fm3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
fm3/ReluRelufm3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
sm1/MatMul/ReadVariableOpReadVariableOp"sm1_matmul_readvariableop_resource*!
_output_shapes
:ò*
dtype0

sm1/MatMulMatMulbm8/Reshape:output:0!sm1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
sm1/BiasAdd/ReadVariableOpReadVariableOp#sm1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
sm1/BiasAddBiasAddsm1/MatMul:product:0"sm1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
sm1/ReluRelusm1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
fm_head/MatMul/ReadVariableOpReadVariableOp&fm_head_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
fm_head/MatMulMatMulfm3/Relu:activations:0%fm_head/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
fm_head/BiasAdd/ReadVariableOpReadVariableOp'fm_head_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
fm_head/BiasAddBiasAddfm_head/MatMul:product:0&fm_head/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
fm_head/SigmoidSigmoidfm_head/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sm_head/MatMul/ReadVariableOpReadVariableOp&sm_head_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
sm_head/MatMulMatMulsm1/Relu:activations:0%sm_head/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sm_head/BiasAdd/ReadVariableOpReadVariableOp'sm_head_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
sm_head/BiasAddBiasAddsm_head/MatMul:product:0&sm_head/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
IdentityIdentitysm_head/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd

Identity_1Identityfm_head/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
NoOpNoOp^bm2/BiasAdd/ReadVariableOp^bm2/Conv2D/ReadVariableOp^bm4/BiasAdd/ReadVariableOp^bm4/Conv2D/ReadVariableOp^bm6/BiasAdd/ReadVariableOp^bm6/Conv2D/ReadVariableOp^fm1/BiasAdd/ReadVariableOp^fm1/MatMul/ReadVariableOp^fm2/BiasAdd/ReadVariableOp^fm2/MatMul/ReadVariableOp^fm3/BiasAdd/ReadVariableOp^fm3/MatMul/ReadVariableOp^fm_head/BiasAdd/ReadVariableOp^fm_head/MatMul/ReadVariableOp^sm1/BiasAdd/ReadVariableOp^sm1/MatMul/ReadVariableOp^sm_head/BiasAdd/ReadVariableOp^sm_head/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ´´: : : : : : : : : : : : : : : : : : 28
bm2/BiasAdd/ReadVariableOpbm2/BiasAdd/ReadVariableOp26
bm2/Conv2D/ReadVariableOpbm2/Conv2D/ReadVariableOp28
bm4/BiasAdd/ReadVariableOpbm4/BiasAdd/ReadVariableOp26
bm4/Conv2D/ReadVariableOpbm4/Conv2D/ReadVariableOp28
bm6/BiasAdd/ReadVariableOpbm6/BiasAdd/ReadVariableOp26
bm6/Conv2D/ReadVariableOpbm6/Conv2D/ReadVariableOp28
fm1/BiasAdd/ReadVariableOpfm1/BiasAdd/ReadVariableOp26
fm1/MatMul/ReadVariableOpfm1/MatMul/ReadVariableOp28
fm2/BiasAdd/ReadVariableOpfm2/BiasAdd/ReadVariableOp26
fm2/MatMul/ReadVariableOpfm2/MatMul/ReadVariableOp28
fm3/BiasAdd/ReadVariableOpfm3/BiasAdd/ReadVariableOp26
fm3/MatMul/ReadVariableOpfm3/MatMul/ReadVariableOp2@
fm_head/BiasAdd/ReadVariableOpfm_head/BiasAdd/ReadVariableOp2>
fm_head/MatMul/ReadVariableOpfm_head/MatMul/ReadVariableOp28
sm1/BiasAdd/ReadVariableOpsm1/BiasAdd/ReadVariableOp26
sm1/MatMul/ReadVariableOpsm1/MatMul/ReadVariableOp2@
sm_head/BiasAdd/ReadVariableOpsm_head/BiasAdd/ReadVariableOp2>
sm_head/MatMul/ReadVariableOpsm_head/MatMul/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs
À

#__inference_fm1_layer_call_fn_87666

inputs
unknown:ò
	unknown_0:	
identity¢StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_fm1_layer_call_and_return_conditional_losses_86729p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿò: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
 
_user_specified_nameinputs

÷
>__inference_bm2_layer_call_and_return_conditional_losses_86667

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ´´: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs
Û
ü
%__inference_model_layer_call_fn_86862
input_1!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@
	unknown_5:ò
	unknown_6:	
	unknown_7:	@
	unknown_8:@
	unknown_9:@ 

unknown_10: 

unknown_11:ò

unknown_12:	

unknown_13: 

unknown_14:

unknown_15:	

unknown_16:
identity

identity_1¢StatefulPartitionedCallÁ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_86821o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ´´: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
!
_user_specified_name	input_1
¥

ó
>__inference_sm1_layer_call_and_return_conditional_losses_87717

inputs3
matmul_readvariableop_resource:ò.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpw
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:ò*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿò: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
 
_user_specified_nameinputs
Ñ
°&
!__inference__traced_restore_88206
file_prefix5
assignvariableop_bm2_kernel:)
assignvariableop_1_bm2_bias:7
assignvariableop_2_bm4_kernel: )
assignvariableop_3_bm4_bias: 7
assignvariableop_4_bm6_kernel: @)
assignvariableop_5_bm6_bias:@2
assignvariableop_6_fm1_kernel:ò*
assignvariableop_7_fm1_bias:	0
assignvariableop_8_fm2_kernel:	@)
assignvariableop_9_fm2_bias:@3
assignvariableop_10_sm1_kernel:ò+
assignvariableop_11_sm1_bias:	0
assignvariableop_12_fm3_kernel:@ *
assignvariableop_13_fm3_bias: 5
"assignvariableop_14_sm_head_kernel:	.
 assignvariableop_15_sm_head_bias:4
"assignvariableop_16_fm_head_kernel: .
 assignvariableop_17_fm_head_bias:'
assignvariableop_18_iteration:	 +
!assignvariableop_19_learning_rate: ?
%assignvariableop_20_adam_m_bm2_kernel:?
%assignvariableop_21_adam_v_bm2_kernel:1
#assignvariableop_22_adam_m_bm2_bias:1
#assignvariableop_23_adam_v_bm2_bias:?
%assignvariableop_24_adam_m_bm4_kernel: ?
%assignvariableop_25_adam_v_bm4_kernel: 1
#assignvariableop_26_adam_m_bm4_bias: 1
#assignvariableop_27_adam_v_bm4_bias: ?
%assignvariableop_28_adam_m_bm6_kernel: @?
%assignvariableop_29_adam_v_bm6_kernel: @1
#assignvariableop_30_adam_m_bm6_bias:@1
#assignvariableop_31_adam_v_bm6_bias:@:
%assignvariableop_32_adam_m_fm1_kernel:ò:
%assignvariableop_33_adam_v_fm1_kernel:ò2
#assignvariableop_34_adam_m_fm1_bias:	2
#assignvariableop_35_adam_v_fm1_bias:	8
%assignvariableop_36_adam_m_fm2_kernel:	@8
%assignvariableop_37_adam_v_fm2_kernel:	@1
#assignvariableop_38_adam_m_fm2_bias:@1
#assignvariableop_39_adam_v_fm2_bias:@:
%assignvariableop_40_adam_m_sm1_kernel:ò:
%assignvariableop_41_adam_v_sm1_kernel:ò2
#assignvariableop_42_adam_m_sm1_bias:	2
#assignvariableop_43_adam_v_sm1_bias:	7
%assignvariableop_44_adam_m_fm3_kernel:@ 7
%assignvariableop_45_adam_v_fm3_kernel:@ 1
#assignvariableop_46_adam_m_fm3_bias: 1
#assignvariableop_47_adam_v_fm3_bias: <
)assignvariableop_48_adam_m_sm_head_kernel:	<
)assignvariableop_49_adam_v_sm_head_kernel:	5
'assignvariableop_50_adam_m_sm_head_bias:5
'assignvariableop_51_adam_v_sm_head_bias:;
)assignvariableop_52_adam_m_fm_head_kernel: ;
)assignvariableop_53_adam_v_fm_head_kernel: 5
'assignvariableop_54_adam_m_fm_head_bias:5
'assignvariableop_55_adam_v_fm_head_bias:%
assignvariableop_56_total_4: %
assignvariableop_57_count_4: %
assignvariableop_58_total_3: %
assignvariableop_59_count_3: %
assignvariableop_60_total_2: %
assignvariableop_61_count_2: %
assignvariableop_62_total_1: %
assignvariableop_63_count_1: #
assignvariableop_64_total: #
assignvariableop_65_count: 
identity_67¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9µ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*Û
valueÑBÎCB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHù
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*
valueBCB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ð
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*¢
_output_shapes
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Q
dtypesG
E2C	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:®
AssignVariableOpAssignVariableOpassignvariableop_bm2_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_1AssignVariableOpassignvariableop_1_bm2_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_2AssignVariableOpassignvariableop_2_bm4_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_3AssignVariableOpassignvariableop_3_bm4_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_4AssignVariableOpassignvariableop_4_bm6_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_5AssignVariableOpassignvariableop_5_bm6_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_6AssignVariableOpassignvariableop_6_fm1_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_7AssignVariableOpassignvariableop_7_fm1_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_8AssignVariableOpassignvariableop_8_fm2_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_9AssignVariableOpassignvariableop_9_fm2_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:·
AssignVariableOp_10AssignVariableOpassignvariableop_10_sm1_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:µ
AssignVariableOp_11AssignVariableOpassignvariableop_11_sm1_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:·
AssignVariableOp_12AssignVariableOpassignvariableop_12_fm3_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:µ
AssignVariableOp_13AssignVariableOpassignvariableop_13_fm3_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:»
AssignVariableOp_14AssignVariableOp"assignvariableop_14_sm_head_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:¹
AssignVariableOp_15AssignVariableOp assignvariableop_15_sm_head_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:»
AssignVariableOp_16AssignVariableOp"assignvariableop_16_fm_head_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:¹
AssignVariableOp_17AssignVariableOp assignvariableop_17_fm_head_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0	*
_output_shapes
:¶
AssignVariableOp_18AssignVariableOpassignvariableop_18_iterationIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:º
AssignVariableOp_19AssignVariableOp!assignvariableop_19_learning_rateIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:¾
AssignVariableOp_20AssignVariableOp%assignvariableop_20_adam_m_bm2_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:¾
AssignVariableOp_21AssignVariableOp%assignvariableop_21_adam_v_bm2_kernelIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:¼
AssignVariableOp_22AssignVariableOp#assignvariableop_22_adam_m_bm2_biasIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:¼
AssignVariableOp_23AssignVariableOp#assignvariableop_23_adam_v_bm2_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:¾
AssignVariableOp_24AssignVariableOp%assignvariableop_24_adam_m_bm4_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:¾
AssignVariableOp_25AssignVariableOp%assignvariableop_25_adam_v_bm4_kernelIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:¼
AssignVariableOp_26AssignVariableOp#assignvariableop_26_adam_m_bm4_biasIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:¼
AssignVariableOp_27AssignVariableOp#assignvariableop_27_adam_v_bm4_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:¾
AssignVariableOp_28AssignVariableOp%assignvariableop_28_adam_m_bm6_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:¾
AssignVariableOp_29AssignVariableOp%assignvariableop_29_adam_v_bm6_kernelIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:¼
AssignVariableOp_30AssignVariableOp#assignvariableop_30_adam_m_bm6_biasIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:¼
AssignVariableOp_31AssignVariableOp#assignvariableop_31_adam_v_bm6_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:¾
AssignVariableOp_32AssignVariableOp%assignvariableop_32_adam_m_fm1_kernelIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:¾
AssignVariableOp_33AssignVariableOp%assignvariableop_33_adam_v_fm1_kernelIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:¼
AssignVariableOp_34AssignVariableOp#assignvariableop_34_adam_m_fm1_biasIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:¼
AssignVariableOp_35AssignVariableOp#assignvariableop_35_adam_v_fm1_biasIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:¾
AssignVariableOp_36AssignVariableOp%assignvariableop_36_adam_m_fm2_kernelIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:¾
AssignVariableOp_37AssignVariableOp%assignvariableop_37_adam_v_fm2_kernelIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:¼
AssignVariableOp_38AssignVariableOp#assignvariableop_38_adam_m_fm2_biasIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:¼
AssignVariableOp_39AssignVariableOp#assignvariableop_39_adam_v_fm2_biasIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:¾
AssignVariableOp_40AssignVariableOp%assignvariableop_40_adam_m_sm1_kernelIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:¾
AssignVariableOp_41AssignVariableOp%assignvariableop_41_adam_v_sm1_kernelIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:¼
AssignVariableOp_42AssignVariableOp#assignvariableop_42_adam_m_sm1_biasIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:¼
AssignVariableOp_43AssignVariableOp#assignvariableop_43_adam_v_sm1_biasIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:¾
AssignVariableOp_44AssignVariableOp%assignvariableop_44_adam_m_fm3_kernelIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:¾
AssignVariableOp_45AssignVariableOp%assignvariableop_45_adam_v_fm3_kernelIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:¼
AssignVariableOp_46AssignVariableOp#assignvariableop_46_adam_m_fm3_biasIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:¼
AssignVariableOp_47AssignVariableOp#assignvariableop_47_adam_v_fm3_biasIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:Â
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_m_sm_head_kernelIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:Â
AssignVariableOp_49AssignVariableOp)assignvariableop_49_adam_v_sm_head_kernelIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:À
AssignVariableOp_50AssignVariableOp'assignvariableop_50_adam_m_sm_head_biasIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:À
AssignVariableOp_51AssignVariableOp'assignvariableop_51_adam_v_sm_head_biasIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:Â
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adam_m_fm_head_kernelIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:Â
AssignVariableOp_53AssignVariableOp)assignvariableop_53_adam_v_fm_head_kernelIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:À
AssignVariableOp_54AssignVariableOp'assignvariableop_54_adam_m_fm_head_biasIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:À
AssignVariableOp_55AssignVariableOp'assignvariableop_55_adam_v_fm_head_biasIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_56AssignVariableOpassignvariableop_56_total_4Identity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_57AssignVariableOpassignvariableop_57_count_4Identity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_58AssignVariableOpassignvariableop_58_total_3Identity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_59AssignVariableOpassignvariableop_59_count_3Identity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_60AssignVariableOpassignvariableop_60_total_2Identity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_61AssignVariableOpassignvariableop_61_count_2Identity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_62AssignVariableOpassignvariableop_62_total_1Identity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_63AssignVariableOpassignvariableop_63_count_1Identity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_64AssignVariableOpassignvariableop_64_totalIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_65AssignVariableOpassignvariableop_65_countIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 û
Identity_66Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_67IdentityIdentity_66:output:0^NoOp_1*
T0*
_output_shapes
: è
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_67Identity_67:output:0*
_input_shapes
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix


ó
B__inference_fm_head_layer_call_and_return_conditional_losses_87776

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ý

÷
>__inference_bm6_layer_call_and_return_conditional_losses_86703

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ-- : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-- 
 
_user_specified_nameinputs
ý

÷
>__inference_bm4_layer_call_and_return_conditional_losses_86685

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿZZ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿZZ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿZZ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿZZ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿZZ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿZZ
 
_user_specified_nameinputs


ï
>__inference_fm3_layer_call_and_return_conditional_losses_87737

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Â
Z
>__inference_bm8_layer_call_and_return_conditional_losses_87657

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ y  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿòZ
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ý

÷
>__inference_bm4_layer_call_and_return_conditional_losses_87606

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿZZ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿZZ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿZZ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿZZ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿZZ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿZZ
 
_user_specified_nameinputs
Þ

#__inference_bm4_layer_call_fn_87595

inputs!
unknown: 
	unknown_0: 
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿZZ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_bm4_layer_call_and_return_conditional_losses_86685w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿZZ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿZZ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿZZ
 
_user_specified_nameinputs
Â
Z
>__inference_bm8_layer_call_and_return_conditional_losses_86716

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ y  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿòZ
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


ï
>__inference_fm3_layer_call_and_return_conditional_losses_86763

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ºs
×
__inference__traced_save_87998
file_prefix)
%savev2_bm2_kernel_read_readvariableop'
#savev2_bm2_bias_read_readvariableop)
%savev2_bm4_kernel_read_readvariableop'
#savev2_bm4_bias_read_readvariableop)
%savev2_bm6_kernel_read_readvariableop'
#savev2_bm6_bias_read_readvariableop)
%savev2_fm1_kernel_read_readvariableop'
#savev2_fm1_bias_read_readvariableop)
%savev2_fm2_kernel_read_readvariableop'
#savev2_fm2_bias_read_readvariableop)
%savev2_sm1_kernel_read_readvariableop'
#savev2_sm1_bias_read_readvariableop)
%savev2_fm3_kernel_read_readvariableop'
#savev2_fm3_bias_read_readvariableop-
)savev2_sm_head_kernel_read_readvariableop+
'savev2_sm_head_bias_read_readvariableop-
)savev2_fm_head_kernel_read_readvariableop+
'savev2_fm_head_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableop0
,savev2_adam_m_bm2_kernel_read_readvariableop0
,savev2_adam_v_bm2_kernel_read_readvariableop.
*savev2_adam_m_bm2_bias_read_readvariableop.
*savev2_adam_v_bm2_bias_read_readvariableop0
,savev2_adam_m_bm4_kernel_read_readvariableop0
,savev2_adam_v_bm4_kernel_read_readvariableop.
*savev2_adam_m_bm4_bias_read_readvariableop.
*savev2_adam_v_bm4_bias_read_readvariableop0
,savev2_adam_m_bm6_kernel_read_readvariableop0
,savev2_adam_v_bm6_kernel_read_readvariableop.
*savev2_adam_m_bm6_bias_read_readvariableop.
*savev2_adam_v_bm6_bias_read_readvariableop0
,savev2_adam_m_fm1_kernel_read_readvariableop0
,savev2_adam_v_fm1_kernel_read_readvariableop.
*savev2_adam_m_fm1_bias_read_readvariableop.
*savev2_adam_v_fm1_bias_read_readvariableop0
,savev2_adam_m_fm2_kernel_read_readvariableop0
,savev2_adam_v_fm2_kernel_read_readvariableop.
*savev2_adam_m_fm2_bias_read_readvariableop.
*savev2_adam_v_fm2_bias_read_readvariableop0
,savev2_adam_m_sm1_kernel_read_readvariableop0
,savev2_adam_v_sm1_kernel_read_readvariableop.
*savev2_adam_m_sm1_bias_read_readvariableop.
*savev2_adam_v_sm1_bias_read_readvariableop0
,savev2_adam_m_fm3_kernel_read_readvariableop0
,savev2_adam_v_fm3_kernel_read_readvariableop.
*savev2_adam_m_fm3_bias_read_readvariableop.
*savev2_adam_v_fm3_bias_read_readvariableop4
0savev2_adam_m_sm_head_kernel_read_readvariableop4
0savev2_adam_v_sm_head_kernel_read_readvariableop2
.savev2_adam_m_sm_head_bias_read_readvariableop2
.savev2_adam_v_sm_head_bias_read_readvariableop4
0savev2_adam_m_fm_head_kernel_read_readvariableop4
0savev2_adam_v_fm_head_kernel_read_readvariableop2
.savev2_adam_m_fm_head_bias_read_readvariableop2
.savev2_adam_v_fm_head_bias_read_readvariableop&
"savev2_total_4_read_readvariableop&
"savev2_count_4_read_readvariableop&
"savev2_total_3_read_readvariableop&
"savev2_count_3_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
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
: ²
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*Û
valueÑBÎCB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHö
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*
valueBCB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B þ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0%savev2_bm2_kernel_read_readvariableop#savev2_bm2_bias_read_readvariableop%savev2_bm4_kernel_read_readvariableop#savev2_bm4_bias_read_readvariableop%savev2_bm6_kernel_read_readvariableop#savev2_bm6_bias_read_readvariableop%savev2_fm1_kernel_read_readvariableop#savev2_fm1_bias_read_readvariableop%savev2_fm2_kernel_read_readvariableop#savev2_fm2_bias_read_readvariableop%savev2_sm1_kernel_read_readvariableop#savev2_sm1_bias_read_readvariableop%savev2_fm3_kernel_read_readvariableop#savev2_fm3_bias_read_readvariableop)savev2_sm_head_kernel_read_readvariableop'savev2_sm_head_bias_read_readvariableop)savev2_fm_head_kernel_read_readvariableop'savev2_fm_head_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop,savev2_adam_m_bm2_kernel_read_readvariableop,savev2_adam_v_bm2_kernel_read_readvariableop*savev2_adam_m_bm2_bias_read_readvariableop*savev2_adam_v_bm2_bias_read_readvariableop,savev2_adam_m_bm4_kernel_read_readvariableop,savev2_adam_v_bm4_kernel_read_readvariableop*savev2_adam_m_bm4_bias_read_readvariableop*savev2_adam_v_bm4_bias_read_readvariableop,savev2_adam_m_bm6_kernel_read_readvariableop,savev2_adam_v_bm6_kernel_read_readvariableop*savev2_adam_m_bm6_bias_read_readvariableop*savev2_adam_v_bm6_bias_read_readvariableop,savev2_adam_m_fm1_kernel_read_readvariableop,savev2_adam_v_fm1_kernel_read_readvariableop*savev2_adam_m_fm1_bias_read_readvariableop*savev2_adam_v_fm1_bias_read_readvariableop,savev2_adam_m_fm2_kernel_read_readvariableop,savev2_adam_v_fm2_kernel_read_readvariableop*savev2_adam_m_fm2_bias_read_readvariableop*savev2_adam_v_fm2_bias_read_readvariableop,savev2_adam_m_sm1_kernel_read_readvariableop,savev2_adam_v_sm1_kernel_read_readvariableop*savev2_adam_m_sm1_bias_read_readvariableop*savev2_adam_v_sm1_bias_read_readvariableop,savev2_adam_m_fm3_kernel_read_readvariableop,savev2_adam_v_fm3_kernel_read_readvariableop*savev2_adam_m_fm3_bias_read_readvariableop*savev2_adam_v_fm3_bias_read_readvariableop0savev2_adam_m_sm_head_kernel_read_readvariableop0savev2_adam_v_sm_head_kernel_read_readvariableop.savev2_adam_m_sm_head_bias_read_readvariableop.savev2_adam_v_sm_head_bias_read_readvariableop0savev2_adam_m_fm_head_kernel_read_readvariableop0savev2_adam_v_fm_head_kernel_read_readvariableop.savev2_adam_m_fm_head_bias_read_readvariableop.savev2_adam_v_fm_head_bias_read_readvariableop"savev2_total_4_read_readvariableop"savev2_count_4_read_readvariableop"savev2_total_3_read_readvariableop"savev2_count_3_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *Q
dtypesG
E2C	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:³
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
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

identity_1Identity_1:output:0*Ç
_input_shapesµ
²: ::: : : @:@:ò::	@:@:ò::@ : :	:: :: : ::::: : : : : @: @:@:@:ò:ò:::	@:	@:@:@:ò:ò:::@ :@ : : :	:	::: : ::: : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:'#
!
_output_shapes
:ò:!

_output_shapes	
::%	!

_output_shapes
:	@: 


_output_shapes
:@:'#
!
_output_shapes
:ò:!

_output_shapes	
::$ 

_output_shapes

:@ : 

_output_shapes
: :%!

_output_shapes
:	: 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @:,(
&
_output_shapes
: @: 

_output_shapes
:@:  

_output_shapes
:@:'!#
!
_output_shapes
:ò:'"#
!
_output_shapes
:ò:!#

_output_shapes	
::!$

_output_shapes	
::%%!

_output_shapes
:	@:%&!

_output_shapes
:	@: '

_output_shapes
:@: (

_output_shapes
:@:')#
!
_output_shapes
:ò:'*#
!
_output_shapes
:ò:!+

_output_shapes	
::!,

_output_shapes	
::$- 

_output_shapes

:@ :$. 

_output_shapes

:@ : /

_output_shapes
: : 0

_output_shapes
: :%1!

_output_shapes
:	:%2!

_output_shapes
:	: 3

_output_shapes
:: 4

_output_shapes
::$5 

_output_shapes

: :$6 

_output_shapes

: : 7

_output_shapes
:: 8

_output_shapes
::9

_output_shapes
: ::

_output_shapes
: :;

_output_shapes
: :<

_output_shapes
: :=

_output_shapes
: :>

_output_shapes
: :?

_output_shapes
: :@

_output_shapes
: :A

_output_shapes
: :B

_output_shapes
: :C

_output_shapes
: 
¹
?
#__inference_bm1_layer_call_fn_87548

inputs
identity³
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_bm1_layer_call_and_return_conditional_losses_86654j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ´´:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs
¾

'__inference_fm_head_layer_call_fn_87765

inputs
unknown: 
	unknown_0:
identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_fm_head_layer_call_and_return_conditional_losses_86797o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
æ

#__inference_bm2_layer_call_fn_87565

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_bm2_layer_call_and_return_conditional_losses_86667y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ´´: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs
¥

ó
>__inference_fm1_layer_call_and_return_conditional_losses_86729

inputs3
matmul_readvariableop_resource:ò.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpw
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:ò*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿò: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
 
_user_specified_nameinputs
¥

ó
>__inference_fm1_layer_call_and_return_conditional_losses_87677

inputs3
matmul_readvariableop_resource:ò.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpw
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:ò*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿò: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
 
_user_specified_nameinputs
¥

ó
>__inference_sm1_layer_call_and_return_conditional_losses_86780

inputs3
matmul_readvariableop_resource:ò.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpw
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:ò*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿò: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
 
_user_specified_nameinputs
R
ã
@__inference_model_layer_call_and_return_conditional_losses_87467

inputs<
"bm2_conv2d_readvariableop_resource:1
#bm2_biasadd_readvariableop_resource:<
"bm4_conv2d_readvariableop_resource: 1
#bm4_biasadd_readvariableop_resource: <
"bm6_conv2d_readvariableop_resource: @1
#bm6_biasadd_readvariableop_resource:@7
"fm1_matmul_readvariableop_resource:ò2
#fm1_biasadd_readvariableop_resource:	5
"fm2_matmul_readvariableop_resource:	@1
#fm2_biasadd_readvariableop_resource:@4
"fm3_matmul_readvariableop_resource:@ 1
#fm3_biasadd_readvariableop_resource: 7
"sm1_matmul_readvariableop_resource:ò2
#sm1_biasadd_readvariableop_resource:	8
&fm_head_matmul_readvariableop_resource: 5
'fm_head_biasadd_readvariableop_resource:9
&sm_head_matmul_readvariableop_resource:	5
'sm_head_biasadd_readvariableop_resource:
identity

identity_1¢bm2/BiasAdd/ReadVariableOp¢bm2/Conv2D/ReadVariableOp¢bm4/BiasAdd/ReadVariableOp¢bm4/Conv2D/ReadVariableOp¢bm6/BiasAdd/ReadVariableOp¢bm6/Conv2D/ReadVariableOp¢fm1/BiasAdd/ReadVariableOp¢fm1/MatMul/ReadVariableOp¢fm2/BiasAdd/ReadVariableOp¢fm2/MatMul/ReadVariableOp¢fm3/BiasAdd/ReadVariableOp¢fm3/MatMul/ReadVariableOp¢fm_head/BiasAdd/ReadVariableOp¢fm_head/MatMul/ReadVariableOp¢sm1/BiasAdd/ReadVariableOp¢sm1/MatMul/ReadVariableOp¢sm_head/BiasAdd/ReadVariableOp¢sm_head/MatMul/ReadVariableOpO

bm1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;Q
bm1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    g
bm1/mulMulinputsbm1/Cast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´p
bm1/addAddV2bm1/mul:z:0bm1/Cast_1/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
bm2/Conv2D/ReadVariableOpReadVariableOp"bm2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0¨

bm2/Conv2DConv2Dbm1/add:z:0!bm2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´*
paddingSAME*
strides
z
bm2/BiasAdd/ReadVariableOpReadVariableOp#bm2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
bm2/BiasAddBiasAddbm2/Conv2D:output:0"bm2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´b
bm2/ReluRelubm2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
bm3/MaxPoolMaxPoolbm2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿZZ*
ksize
*
paddingVALID*
strides

bm4/Conv2D/ReadVariableOpReadVariableOp"bm4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0¯

bm4/Conv2DConv2Dbm3/MaxPool:output:0!bm4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿZZ *
paddingSAME*
strides
z
bm4/BiasAdd/ReadVariableOpReadVariableOp#bm4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
bm4/BiasAddBiasAddbm4/Conv2D:output:0"bm4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿZZ `
bm4/ReluRelubm4/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿZZ 
bm5/MaxPoolMaxPoolbm4/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-- *
ksize
*
paddingVALID*
strides

bm6/Conv2D/ReadVariableOpReadVariableOp"bm6_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0¯

bm6/Conv2DConv2Dbm5/MaxPool:output:0!bm6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--@*
paddingSAME*
strides
z
bm6/BiasAdd/ReadVariableOpReadVariableOp#bm6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
bm6/BiasAddBiasAddbm6/Conv2D:output:0"bm6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--@`
bm6/ReluRelubm6/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--@
bm7/MaxPoolMaxPoolbm6/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
Z
	bm8/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ y  t
bm8/ReshapeReshapebm7/MaxPool:output:0bm8/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
fm1/MatMul/ReadVariableOpReadVariableOp"fm1_matmul_readvariableop_resource*!
_output_shapes
:ò*
dtype0

fm1/MatMulMatMulbm8/Reshape:output:0!fm1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
fm1/BiasAdd/ReadVariableOpReadVariableOp#fm1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
fm1/BiasAddBiasAddfm1/MatMul:product:0"fm1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
fm1/ReluRelufm1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
fm2/MatMul/ReadVariableOpReadVariableOp"fm2_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0

fm2/MatMulMatMulfm1/Relu:activations:0!fm2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@z
fm2/BiasAdd/ReadVariableOpReadVariableOp#fm2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
fm2/BiasAddBiasAddfm2/MatMul:product:0"fm2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@X
fm2/ReluRelufm2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@|
fm3/MatMul/ReadVariableOpReadVariableOp"fm3_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0

fm3/MatMulMatMulfm2/Relu:activations:0!fm3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ z
fm3/BiasAdd/ReadVariableOpReadVariableOp#fm3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
fm3/BiasAddBiasAddfm3/MatMul:product:0"fm3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ X
fm3/ReluRelufm3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
sm1/MatMul/ReadVariableOpReadVariableOp"sm1_matmul_readvariableop_resource*!
_output_shapes
:ò*
dtype0

sm1/MatMulMatMulbm8/Reshape:output:0!sm1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
sm1/BiasAdd/ReadVariableOpReadVariableOp#sm1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
sm1/BiasAddBiasAddsm1/MatMul:product:0"sm1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
sm1/ReluRelusm1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
fm_head/MatMul/ReadVariableOpReadVariableOp&fm_head_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
fm_head/MatMulMatMulfm3/Relu:activations:0%fm_head/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
fm_head/BiasAdd/ReadVariableOpReadVariableOp'fm_head_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
fm_head/BiasAddBiasAddfm_head/MatMul:product:0&fm_head/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
fm_head/SigmoidSigmoidfm_head/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sm_head/MatMul/ReadVariableOpReadVariableOp&sm_head_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
sm_head/MatMulMatMulsm1/Relu:activations:0%sm_head/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sm_head/BiasAdd/ReadVariableOpReadVariableOp'sm_head_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
sm_head/BiasAddBiasAddsm_head/MatMul:product:0&sm_head/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
IdentityIdentitysm_head/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd

Identity_1Identityfm_head/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
NoOpNoOp^bm2/BiasAdd/ReadVariableOp^bm2/Conv2D/ReadVariableOp^bm4/BiasAdd/ReadVariableOp^bm4/Conv2D/ReadVariableOp^bm6/BiasAdd/ReadVariableOp^bm6/Conv2D/ReadVariableOp^fm1/BiasAdd/ReadVariableOp^fm1/MatMul/ReadVariableOp^fm2/BiasAdd/ReadVariableOp^fm2/MatMul/ReadVariableOp^fm3/BiasAdd/ReadVariableOp^fm3/MatMul/ReadVariableOp^fm_head/BiasAdd/ReadVariableOp^fm_head/MatMul/ReadVariableOp^sm1/BiasAdd/ReadVariableOp^sm1/MatMul/ReadVariableOp^sm_head/BiasAdd/ReadVariableOp^sm_head/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ´´: : : : : : : : : : : : : : : : : : 28
bm2/BiasAdd/ReadVariableOpbm2/BiasAdd/ReadVariableOp26
bm2/Conv2D/ReadVariableOpbm2/Conv2D/ReadVariableOp28
bm4/BiasAdd/ReadVariableOpbm4/BiasAdd/ReadVariableOp26
bm4/Conv2D/ReadVariableOpbm4/Conv2D/ReadVariableOp28
bm6/BiasAdd/ReadVariableOpbm6/BiasAdd/ReadVariableOp26
bm6/Conv2D/ReadVariableOpbm6/Conv2D/ReadVariableOp28
fm1/BiasAdd/ReadVariableOpfm1/BiasAdd/ReadVariableOp26
fm1/MatMul/ReadVariableOpfm1/MatMul/ReadVariableOp28
fm2/BiasAdd/ReadVariableOpfm2/BiasAdd/ReadVariableOp26
fm2/MatMul/ReadVariableOpfm2/MatMul/ReadVariableOp28
fm3/BiasAdd/ReadVariableOpfm3/BiasAdd/ReadVariableOp26
fm3/MatMul/ReadVariableOpfm3/MatMul/ReadVariableOp2@
fm_head/BiasAdd/ReadVariableOpfm_head/BiasAdd/ReadVariableOp2>
fm_head/MatMul/ReadVariableOpfm_head/MatMul/ReadVariableOp28
sm1/BiasAdd/ReadVariableOpsm1/BiasAdd/ReadVariableOp26
sm1/MatMul/ReadVariableOpsm1/MatMul/ReadVariableOp2@
sm_head/BiasAdd/ReadVariableOpsm_head/BiasAdd/ReadVariableOp2>
sm_head/MatMul/ReadVariableOpsm_head/MatMul/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs

÷
>__inference_bm2_layer_call_and_return_conditional_losses_87576

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ´´: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
 
_user_specified_nameinputs
\

 __inference__wrapped_model_86603
input_1B
(model_bm2_conv2d_readvariableop_resource:7
)model_bm2_biasadd_readvariableop_resource:B
(model_bm4_conv2d_readvariableop_resource: 7
)model_bm4_biasadd_readvariableop_resource: B
(model_bm6_conv2d_readvariableop_resource: @7
)model_bm6_biasadd_readvariableop_resource:@=
(model_fm1_matmul_readvariableop_resource:ò8
)model_fm1_biasadd_readvariableop_resource:	;
(model_fm2_matmul_readvariableop_resource:	@7
)model_fm2_biasadd_readvariableop_resource:@:
(model_fm3_matmul_readvariableop_resource:@ 7
)model_fm3_biasadd_readvariableop_resource: =
(model_sm1_matmul_readvariableop_resource:ò8
)model_sm1_biasadd_readvariableop_resource:	>
,model_fm_head_matmul_readvariableop_resource: ;
-model_fm_head_biasadd_readvariableop_resource:?
,model_sm_head_matmul_readvariableop_resource:	;
-model_sm_head_biasadd_readvariableop_resource:
identity

identity_1¢ model/bm2/BiasAdd/ReadVariableOp¢model/bm2/Conv2D/ReadVariableOp¢ model/bm4/BiasAdd/ReadVariableOp¢model/bm4/Conv2D/ReadVariableOp¢ model/bm6/BiasAdd/ReadVariableOp¢model/bm6/Conv2D/ReadVariableOp¢ model/fm1/BiasAdd/ReadVariableOp¢model/fm1/MatMul/ReadVariableOp¢ model/fm2/BiasAdd/ReadVariableOp¢model/fm2/MatMul/ReadVariableOp¢ model/fm3/BiasAdd/ReadVariableOp¢model/fm3/MatMul/ReadVariableOp¢$model/fm_head/BiasAdd/ReadVariableOp¢#model/fm_head/MatMul/ReadVariableOp¢ model/sm1/BiasAdd/ReadVariableOp¢model/sm1/MatMul/ReadVariableOp¢$model/sm_head/BiasAdd/ReadVariableOp¢#model/sm_head/MatMul/ReadVariableOpU
model/bm1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;W
model/bm1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    t
model/bm1/mulMulinput_1model/bm1/Cast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
model/bm1/addAddV2model/bm1/mul:z:0model/bm1/Cast_1/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
model/bm2/Conv2D/ReadVariableOpReadVariableOp(model_bm2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0º
model/bm2/Conv2DConv2Dmodel/bm1/add:z:0'model/bm2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´*
paddingSAME*
strides

 model/bm2/BiasAdd/ReadVariableOpReadVariableOp)model_bm2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
model/bm2/BiasAddBiasAddmodel/bm2/Conv2D:output:0(model/bm2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´n
model/bm2/ReluRelumodel/bm2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´§
model/bm3/MaxPoolMaxPoolmodel/bm2/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿZZ*
ksize
*
paddingVALID*
strides

model/bm4/Conv2D/ReadVariableOpReadVariableOp(model_bm4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Á
model/bm4/Conv2DConv2Dmodel/bm3/MaxPool:output:0'model/bm4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿZZ *
paddingSAME*
strides

 model/bm4/BiasAdd/ReadVariableOpReadVariableOp)model_bm4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
model/bm4/BiasAddBiasAddmodel/bm4/Conv2D:output:0(model/bm4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿZZ l
model/bm4/ReluRelumodel/bm4/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿZZ §
model/bm5/MaxPoolMaxPoolmodel/bm4/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ-- *
ksize
*
paddingVALID*
strides

model/bm6/Conv2D/ReadVariableOpReadVariableOp(model_bm6_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Á
model/bm6/Conv2DConv2Dmodel/bm5/MaxPool:output:0'model/bm6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--@*
paddingSAME*
strides

 model/bm6/BiasAdd/ReadVariableOpReadVariableOp)model_bm6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
model/bm6/BiasAddBiasAddmodel/bm6/Conv2D:output:0(model/bm6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--@l
model/bm6/ReluRelumodel/bm6/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ--@§
model/bm7/MaxPoolMaxPoolmodel/bm6/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
`
model/bm8/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ y  
model/bm8/ReshapeReshapemodel/bm7/MaxPool:output:0model/bm8/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò
model/fm1/MatMul/ReadVariableOpReadVariableOp(model_fm1_matmul_readvariableop_resource*!
_output_shapes
:ò*
dtype0
model/fm1/MatMulMatMulmodel/bm8/Reshape:output:0'model/fm1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 model/fm1/BiasAdd/ReadVariableOpReadVariableOp)model_fm1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
model/fm1/BiasAddBiasAddmodel/fm1/MatMul:product:0(model/fm1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
model/fm1/ReluRelumodel/fm1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model/fm2/MatMul/ReadVariableOpReadVariableOp(model_fm2_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype0
model/fm2/MatMulMatMulmodel/fm1/Relu:activations:0'model/fm2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 model/fm2/BiasAdd/ReadVariableOpReadVariableOp)model_fm2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
model/fm2/BiasAddBiasAddmodel/fm2/MatMul:product:0(model/fm2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@d
model/fm2/ReluRelumodel/fm2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
model/fm3/MatMul/ReadVariableOpReadVariableOp(model_fm3_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0
model/fm3/MatMulMatMulmodel/fm2/Relu:activations:0'model/fm3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 model/fm3/BiasAdd/ReadVariableOpReadVariableOp)model_fm3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
model/fm3/BiasAddBiasAddmodel/fm3/MatMul:product:0(model/fm3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ d
model/fm3/ReluRelumodel/fm3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
model/sm1/MatMul/ReadVariableOpReadVariableOp(model_sm1_matmul_readvariableop_resource*!
_output_shapes
:ò*
dtype0
model/sm1/MatMulMatMulmodel/bm8/Reshape:output:0'model/sm1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 model/sm1/BiasAdd/ReadVariableOpReadVariableOp)model_sm1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
model/sm1/BiasAddBiasAddmodel/sm1/MatMul:product:0(model/sm1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
model/sm1/ReluRelumodel/sm1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#model/fm_head/MatMul/ReadVariableOpReadVariableOp,model_fm_head_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
model/fm_head/MatMulMatMulmodel/fm3/Relu:activations:0+model/fm_head/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$model/fm_head/BiasAdd/ReadVariableOpReadVariableOp-model_fm_head_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0 
model/fm_head/BiasAddBiasAddmodel/fm_head/MatMul:product:0,model/fm_head/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
model/fm_head/SigmoidSigmoidmodel/fm_head/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#model/sm_head/MatMul/ReadVariableOpReadVariableOp,model_sm_head_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
model/sm_head/MatMulMatMulmodel/sm1/Relu:activations:0+model/sm_head/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$model/sm_head/BiasAdd/ReadVariableOpReadVariableOp-model_sm_head_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0 
model/sm_head/BiasAddBiasAddmodel/sm_head/MatMul:product:0,model/sm_head/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitymodel/fm_head/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo

Identity_1Identitymodel/sm_head/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
NoOpNoOp!^model/bm2/BiasAdd/ReadVariableOp ^model/bm2/Conv2D/ReadVariableOp!^model/bm4/BiasAdd/ReadVariableOp ^model/bm4/Conv2D/ReadVariableOp!^model/bm6/BiasAdd/ReadVariableOp ^model/bm6/Conv2D/ReadVariableOp!^model/fm1/BiasAdd/ReadVariableOp ^model/fm1/MatMul/ReadVariableOp!^model/fm2/BiasAdd/ReadVariableOp ^model/fm2/MatMul/ReadVariableOp!^model/fm3/BiasAdd/ReadVariableOp ^model/fm3/MatMul/ReadVariableOp%^model/fm_head/BiasAdd/ReadVariableOp$^model/fm_head/MatMul/ReadVariableOp!^model/sm1/BiasAdd/ReadVariableOp ^model/sm1/MatMul/ReadVariableOp%^model/sm_head/BiasAdd/ReadVariableOp$^model/sm_head/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ´´: : : : : : : : : : : : : : : : : : 2D
 model/bm2/BiasAdd/ReadVariableOp model/bm2/BiasAdd/ReadVariableOp2B
model/bm2/Conv2D/ReadVariableOpmodel/bm2/Conv2D/ReadVariableOp2D
 model/bm4/BiasAdd/ReadVariableOp model/bm4/BiasAdd/ReadVariableOp2B
model/bm4/Conv2D/ReadVariableOpmodel/bm4/Conv2D/ReadVariableOp2D
 model/bm6/BiasAdd/ReadVariableOp model/bm6/BiasAdd/ReadVariableOp2B
model/bm6/Conv2D/ReadVariableOpmodel/bm6/Conv2D/ReadVariableOp2D
 model/fm1/BiasAdd/ReadVariableOp model/fm1/BiasAdd/ReadVariableOp2B
model/fm1/MatMul/ReadVariableOpmodel/fm1/MatMul/ReadVariableOp2D
 model/fm2/BiasAdd/ReadVariableOp model/fm2/BiasAdd/ReadVariableOp2B
model/fm2/MatMul/ReadVariableOpmodel/fm2/MatMul/ReadVariableOp2D
 model/fm3/BiasAdd/ReadVariableOp model/fm3/BiasAdd/ReadVariableOp2B
model/fm3/MatMul/ReadVariableOpmodel/fm3/MatMul/ReadVariableOp2L
$model/fm_head/BiasAdd/ReadVariableOp$model/fm_head/BiasAdd/ReadVariableOp2J
#model/fm_head/MatMul/ReadVariableOp#model/fm_head/MatMul/ReadVariableOp2D
 model/sm1/BiasAdd/ReadVariableOp model/sm1/BiasAdd/ReadVariableOp2B
model/sm1/MatMul/ReadVariableOpmodel/sm1/MatMul/ReadVariableOp2L
$model/sm_head/BiasAdd/ReadVariableOp$model/sm_head/BiasAdd/ReadVariableOp2J
#model/sm_head/MatMul/ReadVariableOp#model/sm_head/MatMul/ReadVariableOp:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´´
!
_user_specified_name	input_1

Z
>__inference_bm5_layer_call_and_return_conditional_losses_86624

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Z
>__inference_bm3_layer_call_and_return_conditional_losses_87586

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Z
>__inference_bm7_layer_call_and_return_conditional_losses_86636

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¥
?
#__inference_bm8_layer_call_fn_87651

inputs
identity«
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_bm8_layer_call_and_return_conditional_losses_86716b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿò"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs"
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ñ
serving_defaultÝ
E
input_1:
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ´´;
fm_head0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ;
sm_head0
StatefulPartitionedCall:1ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:åÅ

layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer-8

layer_with_weights-3

layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer_with_weights-6
layer-12
layer_with_weights-7
layer-13
layer_with_weights-8
layer-14
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
loss

signatures"
_tf_keras_network
"
_tf_keras_input_layer
¥
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
Ý
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias
 (_jit_compiled_convolution_op"
_tf_keras_layer
¥
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses"
_tf_keras_layer
Ý
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses

5kernel
6bias
 7_jit_compiled_convolution_op"
_tf_keras_layer
¥
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses"
_tf_keras_layer
Ý
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses

Dkernel
Ebias
 F_jit_compiled_convolution_op"
_tf_keras_layer
¥
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses"
_tf_keras_layer
»
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses

Ykernel
Zbias"
_tf_keras_layer
»
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses

akernel
bbias"
_tf_keras_layer
»
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses

ikernel
jbias"
_tf_keras_layer
»
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses

qkernel
rbias"
_tf_keras_layer
»
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses

ykernel
zbias"
_tf_keras_layer
¾
{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias"
_tf_keras_layer
¨
&0
'1
52
63
D4
E5
Y6
Z7
a8
b9
i10
j11
q12
r13
y14
z15
16
17"
trackable_list_wrapper
¨
&0
'1
52
63
D4
E5
Y6
Z7
a8
b9
i10
j11
q12
r13
y14
z15
16
17"
trackable_list_wrapper
 "
trackable_list_wrapper
Ï
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ñ
trace_0
trace_1
trace_2
trace_32Þ
%__inference_model_layer_call_fn_86862
%__inference_model_layer_call_fn_87348
%__inference_model_layer_call_fn_87391
%__inference_model_layer_call_fn_87148¿
¶²²
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
annotationsª *
 ztrace_0ztrace_1ztrace_2ztrace_3
½
trace_0
trace_1
trace_2
trace_32Ê
@__inference_model_layer_call_and_return_conditional_losses_87467
@__inference_model_layer_call_and_return_conditional_losses_87543
@__inference_model_layer_call_and_return_conditional_losses_87203
@__inference_model_layer_call_and_return_conditional_losses_87258¿
¶²²
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
annotationsª *
 ztrace_0ztrace_1ztrace_2ztrace_3
ËBÈ
 __inference__wrapped_model_86603input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
£

_variables
_iterations
_learning_rate
_index_dict

_momentums
_velocities
_update_step_xla"
experimentalOptimizer
 "
trackable_dict_wrapper
-
serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
é
trace_02Ê
#__inference_bm1_layer_call_fn_87548¢
²
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
annotationsª *
 ztrace_0

trace_02å
>__inference_bm1_layer_call_and_return_conditional_losses_87556¢
²
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
annotationsª *
 ztrace_0
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
 layers
¡metrics
 ¢layer_regularization_losses
£layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
é
¤trace_02Ê
#__inference_bm2_layer_call_fn_87565¢
²
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
annotationsª *
 z¤trace_0

¥trace_02å
>__inference_bm2_layer_call_and_return_conditional_losses_87576¢
²
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
annotationsª *
 z¥trace_0
$:"2
bm2/kernel
:2bm2/bias
´2±®
£²
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
annotationsª *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
¦non_trainable_variables
§layers
¨metrics
 ©layer_regularization_losses
ªlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
é
«trace_02Ê
#__inference_bm3_layer_call_fn_87581¢
²
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
annotationsª *
 z«trace_0

¬trace_02å
>__inference_bm3_layer_call_and_return_conditional_losses_87586¢
²
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
annotationsª *
 z¬trace_0
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
²
­non_trainable_variables
®layers
¯metrics
 °layer_regularization_losses
±layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
é
²trace_02Ê
#__inference_bm4_layer_call_fn_87595¢
²
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
annotationsª *
 z²trace_0

³trace_02å
>__inference_bm4_layer_call_and_return_conditional_losses_87606¢
²
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
annotationsª *
 z³trace_0
$:" 2
bm4/kernel
: 2bm4/bias
´2±®
£²
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
annotationsª *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
´non_trainable_variables
µlayers
¶metrics
 ·layer_regularization_losses
¸layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
é
¹trace_02Ê
#__inference_bm5_layer_call_fn_87611¢
²
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
annotationsª *
 z¹trace_0

ºtrace_02å
>__inference_bm5_layer_call_and_return_conditional_losses_87616¢
²
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
annotationsª *
 zºtrace_0
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
»non_trainable_variables
¼layers
½metrics
 ¾layer_regularization_losses
¿layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
é
Àtrace_02Ê
#__inference_bm6_layer_call_fn_87625¢
²
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
annotationsª *
 zÀtrace_0

Átrace_02å
>__inference_bm6_layer_call_and_return_conditional_losses_87636¢
²
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
annotationsª *
 zÁtrace_0
$:" @2
bm6/kernel
:@2bm6/bias
´2±®
£²
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
annotationsª *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ânon_trainable_variables
Ãlayers
Ämetrics
 Ålayer_regularization_losses
Ælayer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
é
Çtrace_02Ê
#__inference_bm7_layer_call_fn_87641¢
²
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
annotationsª *
 zÇtrace_0

Ètrace_02å
>__inference_bm7_layer_call_and_return_conditional_losses_87646¢
²
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
annotationsª *
 zÈtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Énon_trainable_variables
Êlayers
Ëmetrics
 Ìlayer_regularization_losses
Ílayer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
é
Îtrace_02Ê
#__inference_bm8_layer_call_fn_87651¢
²
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
annotationsª *
 zÎtrace_0

Ïtrace_02å
>__inference_bm8_layer_call_and_return_conditional_losses_87657¢
²
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
annotationsª *
 zÏtrace_0
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ðnon_trainable_variables
Ñlayers
Òmetrics
 Ólayer_regularization_losses
Ôlayer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
é
Õtrace_02Ê
#__inference_fm1_layer_call_fn_87666¢
²
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
annotationsª *
 zÕtrace_0

Ötrace_02å
>__inference_fm1_layer_call_and_return_conditional_losses_87677¢
²
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
annotationsª *
 zÖtrace_0
:ò2
fm1/kernel
:2fm1/bias
.
a0
b1"
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
×non_trainable_variables
Ølayers
Ùmetrics
 Úlayer_regularization_losses
Ûlayer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
é
Ütrace_02Ê
#__inference_fm2_layer_call_fn_87686¢
²
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
annotationsª *
 zÜtrace_0

Ýtrace_02å
>__inference_fm2_layer_call_and_return_conditional_losses_87697¢
²
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
annotationsª *
 zÝtrace_0
:	@2
fm2/kernel
:@2fm2/bias
.
i0
j1"
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Þnon_trainable_variables
ßlayers
àmetrics
 álayer_regularization_losses
âlayer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
é
ãtrace_02Ê
#__inference_sm1_layer_call_fn_87706¢
²
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
annotationsª *
 zãtrace_0

ätrace_02å
>__inference_sm1_layer_call_and_return_conditional_losses_87717¢
²
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
annotationsª *
 zätrace_0
:ò2
sm1/kernel
:2sm1/bias
.
q0
r1"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ånon_trainable_variables
ælayers
çmetrics
 èlayer_regularization_losses
élayer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
é
êtrace_02Ê
#__inference_fm3_layer_call_fn_87726¢
²
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
annotationsª *
 zêtrace_0

ëtrace_02å
>__inference_fm3_layer_call_and_return_conditional_losses_87737¢
²
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
annotationsª *
 zëtrace_0
:@ 2
fm3/kernel
: 2fm3/bias
.
y0
z1"
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ìnon_trainable_variables
ílayers
îmetrics
 ïlayer_regularization_losses
ðlayer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
í
ñtrace_02Î
'__inference_sm_head_layer_call_fn_87746¢
²
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
annotationsª *
 zñtrace_0

òtrace_02é
B__inference_sm_head_layer_call_and_return_conditional_losses_87756¢
²
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
annotationsª *
 zòtrace_0
!:	2sm_head/kernel
:2sm_head/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
´
ónon_trainable_variables
ôlayers
õmetrics
 ölayer_regularization_losses
÷layer_metrics
{	variables
|trainable_variables
}regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
í
øtrace_02Î
'__inference_fm_head_layer_call_fn_87765¢
²
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
annotationsª *
 zøtrace_0

ùtrace_02é
B__inference_fm_head_layer_call_and_return_conditional_losses_87776¢
²
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
annotationsª *
 zùtrace_0
 : 2fm_head/kernel
:2fm_head/bias
 "
trackable_list_wrapper

0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14"
trackable_list_wrapper
H
ú0
û1
ü2
ý3
þ4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
÷Bô
%__inference_model_layer_call_fn_86862input_1"¿
¶²²
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
annotationsª *
 
öBó
%__inference_model_layer_call_fn_87348inputs"¿
¶²²
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
annotationsª *
 
öBó
%__inference_model_layer_call_fn_87391inputs"¿
¶²²
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
annotationsª *
 
÷Bô
%__inference_model_layer_call_fn_87148input_1"¿
¶²²
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
annotationsª *
 
B
@__inference_model_layer_call_and_return_conditional_losses_87467inputs"¿
¶²²
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
annotationsª *
 
B
@__inference_model_layer_call_and_return_conditional_losses_87543inputs"¿
¶²²
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
annotationsª *
 
B
@__inference_model_layer_call_and_return_conditional_losses_87203input_1"¿
¶²²
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
annotationsª *
 
B
@__inference_model_layer_call_and_return_conditional_losses_87258input_1"¿
¶²²
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
annotationsª *
 
ã
0
ÿ1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
 34
¡35
¢36"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
¸
ÿ0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
¡17"
trackable_list_wrapper
¸
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
 16
¢17"
trackable_list_wrapper
¿2¼¹
®²ª
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
ÊBÇ
#__inference_signature_wrapper_87305input_1"
²
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
annotationsª *
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
×BÔ
#__inference_bm1_layer_call_fn_87548inputs"¢
²
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
annotationsª *
 
òBï
>__inference_bm1_layer_call_and_return_conditional_losses_87556inputs"¢
²
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
annotationsª *
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
×BÔ
#__inference_bm2_layer_call_fn_87565inputs"¢
²
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
annotationsª *
 
òBï
>__inference_bm2_layer_call_and_return_conditional_losses_87576inputs"¢
²
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
annotationsª *
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
×BÔ
#__inference_bm3_layer_call_fn_87581inputs"¢
²
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
annotationsª *
 
òBï
>__inference_bm3_layer_call_and_return_conditional_losses_87586inputs"¢
²
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
annotationsª *
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
×BÔ
#__inference_bm4_layer_call_fn_87595inputs"¢
²
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
annotationsª *
 
òBï
>__inference_bm4_layer_call_and_return_conditional_losses_87606inputs"¢
²
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
annotationsª *
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
×BÔ
#__inference_bm5_layer_call_fn_87611inputs"¢
²
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
annotationsª *
 
òBï
>__inference_bm5_layer_call_and_return_conditional_losses_87616inputs"¢
²
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
annotationsª *
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
×BÔ
#__inference_bm6_layer_call_fn_87625inputs"¢
²
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
annotationsª *
 
òBï
>__inference_bm6_layer_call_and_return_conditional_losses_87636inputs"¢
²
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
annotationsª *
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
×BÔ
#__inference_bm7_layer_call_fn_87641inputs"¢
²
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
annotationsª *
 
òBï
>__inference_bm7_layer_call_and_return_conditional_losses_87646inputs"¢
²
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
annotationsª *
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
×BÔ
#__inference_bm8_layer_call_fn_87651inputs"¢
²
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
annotationsª *
 
òBï
>__inference_bm8_layer_call_and_return_conditional_losses_87657inputs"¢
²
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
annotationsª *
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
×BÔ
#__inference_fm1_layer_call_fn_87666inputs"¢
²
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
annotationsª *
 
òBï
>__inference_fm1_layer_call_and_return_conditional_losses_87677inputs"¢
²
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
annotationsª *
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
×BÔ
#__inference_fm2_layer_call_fn_87686inputs"¢
²
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
annotationsª *
 
òBï
>__inference_fm2_layer_call_and_return_conditional_losses_87697inputs"¢
²
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
annotationsª *
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
×BÔ
#__inference_sm1_layer_call_fn_87706inputs"¢
²
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
annotationsª *
 
òBï
>__inference_sm1_layer_call_and_return_conditional_losses_87717inputs"¢
²
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
annotationsª *
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
×BÔ
#__inference_fm3_layer_call_fn_87726inputs"¢
²
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
annotationsª *
 
òBï
>__inference_fm3_layer_call_and_return_conditional_losses_87737inputs"¢
²
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
annotationsª *
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
ÛBØ
'__inference_sm_head_layer_call_fn_87746inputs"¢
²
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
annotationsª *
 
öBó
B__inference_sm_head_layer_call_and_return_conditional_losses_87756inputs"¢
²
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
annotationsª *
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
ÛBØ
'__inference_fm_head_layer_call_fn_87765inputs"¢
²
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
annotationsª *
 
öBó
B__inference_fm_head_layer_call_and_return_conditional_losses_87776inputs"¢
²
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
annotationsª *
 
R
£	variables
¤	keras_api

¥total

¦count"
_tf_keras_metric
R
§	variables
¨	keras_api

©total

ªcount"
_tf_keras_metric
R
«	variables
¬	keras_api

­total

®count"
_tf_keras_metric
c
¯	variables
°	keras_api

±total

²count
³
_fn_kwargs"
_tf_keras_metric
c
´	variables
µ	keras_api

¶total

·count
¸
_fn_kwargs"
_tf_keras_metric
):'2Adam/m/bm2/kernel
):'2Adam/v/bm2/kernel
:2Adam/m/bm2/bias
:2Adam/v/bm2/bias
):' 2Adam/m/bm4/kernel
):' 2Adam/v/bm4/kernel
: 2Adam/m/bm4/bias
: 2Adam/v/bm4/bias
):' @2Adam/m/bm6/kernel
):' @2Adam/v/bm6/kernel
:@2Adam/m/bm6/bias
:@2Adam/v/bm6/bias
$:"ò2Adam/m/fm1/kernel
$:"ò2Adam/v/fm1/kernel
:2Adam/m/fm1/bias
:2Adam/v/fm1/bias
": 	@2Adam/m/fm2/kernel
": 	@2Adam/v/fm2/kernel
:@2Adam/m/fm2/bias
:@2Adam/v/fm2/bias
$:"ò2Adam/m/sm1/kernel
$:"ò2Adam/v/sm1/kernel
:2Adam/m/sm1/bias
:2Adam/v/sm1/bias
!:@ 2Adam/m/fm3/kernel
!:@ 2Adam/v/fm3/kernel
: 2Adam/m/fm3/bias
: 2Adam/v/fm3/bias
&:$	2Adam/m/sm_head/kernel
&:$	2Adam/v/sm_head/kernel
:2Adam/m/sm_head/bias
:2Adam/v/sm_head/bias
%:# 2Adam/m/fm_head/kernel
%:# 2Adam/v/fm_head/kernel
:2Adam/m/fm_head/bias
:2Adam/v/fm_head/bias
0
¥0
¦1"
trackable_list_wrapper
.
£	variables"
_generic_user_object
:  (2total
:  (2count
0
©0
ª1"
trackable_list_wrapper
.
§	variables"
_generic_user_object
:  (2total
:  (2count
0
­0
®1"
trackable_list_wrapper
.
«	variables"
_generic_user_object
:  (2total
:  (2count
0
±0
²1"
trackable_list_wrapper
.
¯	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
¶0
·1"
trackable_list_wrapper
.
´	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapperØ
 __inference__wrapped_model_86603³&'56DEYZabqrijyz:¢7
0¢-
+(
input_1ÿÿÿÿÿÿÿÿÿ´´
ª "_ª\
,
fm_head!
fm_headÿÿÿÿÿÿÿÿÿ
,
sm_head!
sm_headÿÿÿÿÿÿÿÿÿµ
>__inference_bm1_layer_call_and_return_conditional_losses_87556s9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ´´
ª "6¢3
,)
tensor_0ÿÿÿÿÿÿÿÿÿ´´
 
#__inference_bm1_layer_call_fn_87548h9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ´´
ª "+(
unknownÿÿÿÿÿÿÿÿÿ´´¹
>__inference_bm2_layer_call_and_return_conditional_losses_87576w&'9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ´´
ª "6¢3
,)
tensor_0ÿÿÿÿÿÿÿÿÿ´´
 
#__inference_bm2_layer_call_fn_87565l&'9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ´´
ª "+(
unknownÿÿÿÿÿÿÿÿÿ´´è
>__inference_bm3_layer_call_and_return_conditional_losses_87586¥R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "O¢L
EB
tensor_04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Â
#__inference_bm3_layer_call_fn_87581R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "DA
unknown4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿµ
>__inference_bm4_layer_call_and_return_conditional_losses_87606s567¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿZZ
ª "4¢1
*'
tensor_0ÿÿÿÿÿÿÿÿÿZZ 
 
#__inference_bm4_layer_call_fn_87595h567¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿZZ
ª ")&
unknownÿÿÿÿÿÿÿÿÿZZ è
>__inference_bm5_layer_call_and_return_conditional_losses_87616¥R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "O¢L
EB
tensor_04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Â
#__inference_bm5_layer_call_fn_87611R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "DA
unknown4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿµ
>__inference_bm6_layer_call_and_return_conditional_losses_87636sDE7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ-- 
ª "4¢1
*'
tensor_0ÿÿÿÿÿÿÿÿÿ--@
 
#__inference_bm6_layer_call_fn_87625hDE7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ-- 
ª ")&
unknownÿÿÿÿÿÿÿÿÿ--@è
>__inference_bm7_layer_call_and_return_conditional_losses_87646¥R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "O¢L
EB
tensor_04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Â
#__inference_bm7_layer_call_fn_87641R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "DA
unknown4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ«
>__inference_bm8_layer_call_and_return_conditional_losses_87657i7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª ".¢+
$!
tensor_0ÿÿÿÿÿÿÿÿÿò
 
#__inference_bm8_layer_call_fn_87651^7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "# 
unknownÿÿÿÿÿÿÿÿÿò¨
>__inference_fm1_layer_call_and_return_conditional_losses_87677fYZ1¢.
'¢$
"
inputsÿÿÿÿÿÿÿÿÿò
ª "-¢*
# 
tensor_0ÿÿÿÿÿÿÿÿÿ
 
#__inference_fm1_layer_call_fn_87666[YZ1¢.
'¢$
"
inputsÿÿÿÿÿÿÿÿÿò
ª ""
unknownÿÿÿÿÿÿÿÿÿ¦
>__inference_fm2_layer_call_and_return_conditional_losses_87697dab0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ@
 
#__inference_fm2_layer_call_fn_87686Yab0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "!
unknownÿÿÿÿÿÿÿÿÿ@¥
>__inference_fm3_layer_call_and_return_conditional_losses_87737cqr/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ 
 
#__inference_fm3_layer_call_fn_87726Xqr/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "!
unknownÿÿÿÿÿÿÿÿÿ «
B__inference_fm_head_layer_call_and_return_conditional_losses_87776e/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 
'__inference_fm_head_layer_call_fn_87765Z/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "!
unknownÿÿÿÿÿÿÿÿÿú
@__inference_model_layer_call_and_return_conditional_losses_87203µ&'56DEYZabqrijyzB¢?
8¢5
+(
input_1ÿÿÿÿÿÿÿÿÿ´´
p 

 
ª "Y¢V
OL
$!

tensor_0_0ÿÿÿÿÿÿÿÿÿ
$!

tensor_0_1ÿÿÿÿÿÿÿÿÿ
 ú
@__inference_model_layer_call_and_return_conditional_losses_87258µ&'56DEYZabqrijyzB¢?
8¢5
+(
input_1ÿÿÿÿÿÿÿÿÿ´´
p

 
ª "Y¢V
OL
$!

tensor_0_0ÿÿÿÿÿÿÿÿÿ
$!

tensor_0_1ÿÿÿÿÿÿÿÿÿ
 ù
@__inference_model_layer_call_and_return_conditional_losses_87467´&'56DEYZabqrijyzA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ´´
p 

 
ª "Y¢V
OL
$!

tensor_0_0ÿÿÿÿÿÿÿÿÿ
$!

tensor_0_1ÿÿÿÿÿÿÿÿÿ
 ù
@__inference_model_layer_call_and_return_conditional_losses_87543´&'56DEYZabqrijyzA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ´´
p

 
ª "Y¢V
OL
$!

tensor_0_0ÿÿÿÿÿÿÿÿÿ
$!

tensor_0_1ÿÿÿÿÿÿÿÿÿ
 Ñ
%__inference_model_layer_call_fn_86862§&'56DEYZabqrijyzB¢?
8¢5
+(
input_1ÿÿÿÿÿÿÿÿÿ´´
p 

 
ª "KH
"
tensor_0ÿÿÿÿÿÿÿÿÿ
"
tensor_1ÿÿÿÿÿÿÿÿÿÑ
%__inference_model_layer_call_fn_87148§&'56DEYZabqrijyzB¢?
8¢5
+(
input_1ÿÿÿÿÿÿÿÿÿ´´
p

 
ª "KH
"
tensor_0ÿÿÿÿÿÿÿÿÿ
"
tensor_1ÿÿÿÿÿÿÿÿÿÐ
%__inference_model_layer_call_fn_87348¦&'56DEYZabqrijyzA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ´´
p 

 
ª "KH
"
tensor_0ÿÿÿÿÿÿÿÿÿ
"
tensor_1ÿÿÿÿÿÿÿÿÿÐ
%__inference_model_layer_call_fn_87391¦&'56DEYZabqrijyzA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ´´
p

 
ª "KH
"
tensor_0ÿÿÿÿÿÿÿÿÿ
"
tensor_1ÿÿÿÿÿÿÿÿÿæ
#__inference_signature_wrapper_87305¾&'56DEYZabqrijyzE¢B
¢ 
;ª8
6
input_1+(
input_1ÿÿÿÿÿÿÿÿÿ´´"_ª\
,
fm_head!
fm_headÿÿÿÿÿÿÿÿÿ
,
sm_head!
sm_headÿÿÿÿÿÿÿÿÿ¨
>__inference_sm1_layer_call_and_return_conditional_losses_87717fij1¢.
'¢$
"
inputsÿÿÿÿÿÿÿÿÿò
ª "-¢*
# 
tensor_0ÿÿÿÿÿÿÿÿÿ
 
#__inference_sm1_layer_call_fn_87706[ij1¢.
'¢$
"
inputsÿÿÿÿÿÿÿÿÿò
ª ""
unknownÿÿÿÿÿÿÿÿÿª
B__inference_sm_head_layer_call_and_return_conditional_losses_87756dyz0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 
'__inference_sm_head_layer_call_fn_87746Yyz0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "!
unknownÿÿÿÿÿÿÿÿÿ