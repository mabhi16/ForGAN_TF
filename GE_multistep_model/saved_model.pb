??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
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
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.1.02unknown8??
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:((*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:((*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:(*
dtype0
|
dense_1_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*!
shared_namedense_1_1/kernel
u
$dense_1_1/kernel/Read/ReadVariableOpReadVariableOpdense_1_1/kernel*
_output_shapes

:(*
dtype0
t
dense_1_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1_1/bias
m
"dense_1_1/bias/Read/ReadVariableOpReadVariableOpdense_1_1/bias*
_output_shapes
:*
dtype0
t
gru_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namegru_1/kernel
m
 gru_1/kernel/Read/ReadVariableOpReadVariableOpgru_1/kernel*
_output_shapes

:*
dtype0
?
gru_1/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_namegru_1/recurrent_kernel
?
*gru_1/recurrent_kernel/Read/ReadVariableOpReadVariableOpgru_1/recurrent_kernel*
_output_shapes

:*
dtype0
l

gru_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
gru_1/bias
e
gru_1/bias/Read/ReadVariableOpReadVariableOp
gru_1/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	trainable_variables

regularization_losses
	variables
	keras_api

signatures
 
l
cell

state_spec
trainable_variables
regularization_losses
	variables
	keras_api
 
R
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
 	variables
!	keras_api
h

"kernel
#bias
$trainable_variables
%regularization_losses
&	variables
'	keras_api
R
(trainable_variables
)regularization_losses
*	variables
+	keras_api
1
,0
-1
.2
3
4
"5
#6
 
1
,0
-1
.2
3
4
"5
#6
?
/metrics
0non_trainable_variables
	trainable_variables

regularization_losses

1layers
2layer_regularization_losses
	variables
 
~

,kernel
-recurrent_kernel
.bias
3trainable_variables
4regularization_losses
5	variables
6	keras_api
 

,0
-1
.2
 

,0
-1
.2
?
7metrics
8non_trainable_variables
trainable_variables
regularization_losses

9layers
:layer_regularization_losses
	variables
 
 
 
?
;metrics
<non_trainable_variables
trainable_variables
regularization_losses

=layers
>layer_regularization_losses
	variables
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
?metrics
@non_trainable_variables
trainable_variables
regularization_losses

Alayers
Blayer_regularization_losses
	variables
 
 
 
?
Cmetrics
Dnon_trainable_variables
trainable_variables
regularization_losses

Elayers
Flayer_regularization_losses
 	variables
\Z
VARIABLE_VALUEdense_1_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_1_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

"0
#1
 

"0
#1
?
Gmetrics
Hnon_trainable_variables
$trainable_variables
%regularization_losses

Ilayers
Jlayer_regularization_losses
&	variables
 
 
 
?
Kmetrics
Lnon_trainable_variables
(trainable_variables
)regularization_losses

Mlayers
Nlayer_regularization_losses
*	variables
RP
VARIABLE_VALUEgru_1/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEgru_1/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUE
gru_1/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
 
 
8
0
1
2
3
4
5
6
7
 

,0
-1
.2
 

,0
-1
.2
?
Ometrics
Pnon_trainable_variables
3trainable_variables
4regularization_losses

Qlayers
Rlayer_regularization_losses
5	variables
 
 

0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
?
serving_default_input_1Placeholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
z
serving_default_input_2Placeholder*'
_output_shapes
:????????? *
dtype0*
shape:????????? 
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2gru_1/kernel
gru_1/biasgru_1/recurrent_kerneldense_2/kerneldense_2/biasdense_1_1/kerneldense_1_1/bias*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*-
f(R&
$__inference_signature_wrapper_679309
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp$dense_1_1/kernel/Read/ReadVariableOp"dense_1_1/bias/Read/ReadVariableOp gru_1/kernel/Read/ReadVariableOp*gru_1/recurrent_kernel/Read/ReadVariableOpgru_1/bias/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

CPU

GPU 2J 8*(
f#R!
__inference__traced_save_681274
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_2/kerneldense_2/biasdense_1_1/kerneldense_1_1/biasgru_1/kernelgru_1/recurrent_kernel
gru_1/bias*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

CPU

GPU 2J 8*+
f&R$
"__inference__traced_restore_681307??
?
?
&__inference_dense_layer_call_fn_680983

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????(**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_6791452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????(::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?a
?
D__inference_gru_cell_layer_call_and_return_conditional_losses_678172

inputs

states
readvariableop_resource
readvariableop_3_resource
readvariableop_6_resource
identity

identity_1??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?ReadVariableOp_4?ReadVariableOp_5?ReadVariableOp_6?ReadVariableOp_7?ReadVariableOp_8x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slicel
MatMulMatMulinputsstrided_slice:output:0*
T0*'
_output_shapes
:?????????2
MatMul?
ReadVariableOp_1ReadVariableOpreadvariableop_resource^ReadVariableOp*
_output_shapes

:*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_1r
MatMul_1MatMulinputsstrided_slice_1:output:0*
T0*'
_output_shapes
:?????????2

MatMul_1?
ReadVariableOp_2ReadVariableOpreadvariableop_resource^ReadVariableOp_1*
_output_shapes

:*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_2r
MatMul_2MatMulinputsstrided_slice_2:output:0*
T0*'
_output_shapes
:?????????2

MatMul_2z
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype02
ReadVariableOp_3x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slice_3{
BiasAddBiasAddMatMul:product:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource^ReadVariableOp_3*
_output_shapes
:*
dtype02
ReadVariableOp_4x
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack|
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2?
strided_slice_4StridedSliceReadVariableOp_4:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_4?
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_1?
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource^ReadVariableOp_4*
_output_shapes
:*
dtype02
ReadVariableOp_5x
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack|
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_5/stack_1|
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_2?
strided_slice_5StridedSliceReadVariableOp_5:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
strided_slice_5?
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_2~
ReadVariableOp_6ReadVariableOpreadvariableop_6_resource*
_output_shapes

:*
dtype02
ReadVariableOp_6
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_6:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_6r
MatMul_3MatMulstatesstrided_slice_6:output:0*
T0*'
_output_shapes
:?????????2

MatMul_3?
ReadVariableOp_7ReadVariableOpreadvariableop_6_resource^ReadVariableOp_6*
_output_shapes

:*
dtype02
ReadVariableOp_7
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSliceReadVariableOp_7:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_7r
MatMul_4MatMulstatesstrided_slice_7:output:0*
T0*'
_output_shapes
:?????????2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1\
MulMuladd:z:0Const:output:0*
T0*'
_output_shapes
:?????????2
Mulb
Add_1AddMul:z:0Const_1:output:0*
T0*'
_output_shapes
:?????????2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_valueq
add_2AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3d
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*'
_output_shapes
:?????????2
Mul_1d
Add_3Add	Mul_1:z:0Const_3:output:0*
T0*'
_output_shapes
:?????????2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1d
mul_2Mulclip_by_value_1:z:0states*
T0*'
_output_shapes
:?????????2
mul_2?
ReadVariableOp_8ReadVariableOpreadvariableop_6_resource^ReadVariableOp_7*
_output_shapes

:*
dtype02
ReadVariableOp_8
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack?
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_8/stack_1?
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2?
strided_slice_8StridedSliceReadVariableOp_8:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_8u
MatMul_5MatMul	mul_2:z:0strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2

MatMul_5q
add_4AddV2BiasAdd_2:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
add_4Q
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:?????????2
Tanhb
mul_3Mulclip_by_value:z:0states*
T0*'
_output_shapes
:?????????2
mul_3S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xf
subSubsub/x:output:0clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
subZ
mul_4Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_4_
add_5AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:?????????2
add_5?
IdentityIdentity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:?????????:?????????:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_8:& "
 
_user_specified_nameinputs:&"
 
_user_specified_namestates
?

?
$__inference_signature_wrapper_679309
input_1
input_2"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2input_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8**
f%R#
!__inference__wrapped_model_6779622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????:????????? :::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1:'#
!
_user_specified_name	input_2
?
?
while_cond_680014
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1.
*while_cond_680014___redundant_placeholder0.
*while_cond_680014___redundant_placeholder1.
*while_cond_680014___redundant_placeholder2.
*while_cond_680014___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-: : : : :?????????: ::::
?
?
while_cond_680798
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1.
*while_cond_680798___redundant_placeholder0.
*while_cond_680798___redundant_placeholder1.
*while_cond_680798___redundant_placeholder2.
*while_cond_680798___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-: : : : :?????????: ::::
?
q
G__inference_concatenate_layer_call_and_return_conditional_losses_679126

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????(2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:????????? :& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
?
?
A__inference_model_layer_call_and_return_conditional_losses_679232
input_2
input_1&
"gru_statefulpartitionedcall_args_1&
"gru_statefulpartitionedcall_args_2&
"gru_statefulpartitionedcall_args_3(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?gru/StatefulPartitionedCall?
gru/StatefulPartitionedCallStatefulPartitionedCallinput_1"gru_statefulpartitionedcall_args_1"gru_statefulpartitionedcall_args_2"gru_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_6790982
gru/StatefulPartitionedCall?
concatenate/PartitionedCallPartitionedCall$gru/StatefulPartitionedCall:output:0input_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????(**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_6791262
concatenate/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????(**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_6791452
dense/StatefulPartitionedCall?
re_lu/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????(**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_6791622
re_lu/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_6791802!
dense_1/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_6792052
reshape/PartitionedCall?
IdentityIdentity reshape/PartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^gru/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:????????? :?????????:::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall:' #
!
_user_specified_name	input_2:'#
!
_user_specified_name	input_1
?
?
A__inference_model_layer_call_and_return_conditional_losses_679254

inputs
inputs_1&
"gru_statefulpartitionedcall_args_1&
"gru_statefulpartitionedcall_args_2&
"gru_statefulpartitionedcall_args_3(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?gru/StatefulPartitionedCall?
gru/StatefulPartitionedCallStatefulPartitionedCallinputs_1"gru_statefulpartitionedcall_args_1"gru_statefulpartitionedcall_args_2"gru_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_6788422
gru/StatefulPartitionedCall?
concatenate/PartitionedCallPartitionedCall$gru/StatefulPartitionedCall:output:0inputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????(**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_6791262
concatenate/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????(**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_6791452
dense/StatefulPartitionedCall?
re_lu/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????(**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_6791622
re_lu/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_6791802!
dense_1/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_6792052
reshape/PartitionedCall?
IdentityIdentity reshape/PartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^gru/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:????????? :?????????:::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
?
?
while_cond_678407
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1.
*while_cond_678407___redundant_placeholder0.
*while_cond_678407___redundant_placeholder1.
*while_cond_678407___redundant_placeholder2.
*while_cond_678407___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-: : : : :?????????: ::::
?	
?
)__inference_gru_cell_layer_call_fn_681217

inputs
states_0"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*:
_output_shapes(
&:?????????:?????????**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_gru_cell_layer_call_and_return_conditional_losses_6780832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:?????????:?????????:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0
?v
?
while_body_680015
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
readvariableop_resource_0
readvariableop_3_resource_0
readvariableop_6_resource_0
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
readvariableop_resource
readvariableop_3_resource
readvariableop_6_resource??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?ReadVariableOp_4?ReadVariableOp_5?ReadVariableOp_6?ReadVariableOp_7?ReadVariableOp_8?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemz
ReadVariableOpReadVariableOpreadvariableop_resource_0*
_output_shapes

:*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice?
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice:output:0*
T0*'
_output_shapes
:?????????2
MatMul?
ReadVariableOp_1ReadVariableOpreadvariableop_resource_0^ReadVariableOp*
_output_shapes

:*
dtype02
ReadVariableOp_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_1:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_2?
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2

MatMul_1?
ReadVariableOp_2ReadVariableOpreadvariableop_resource_0^ReadVariableOp_1*
_output_shapes

:*
dtype02
ReadVariableOp_2
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_2:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_3?
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2

MatMul_2|
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource_0*
_output_shapes
:*
dtype02
ReadVariableOp_3x
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_4/stack|
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2?
strided_slice_4StridedSliceReadVariableOp_3:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slice_4{
BiasAddBiasAddMatMul:product:0strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource_0^ReadVariableOp_3*
_output_shapes
:*
dtype02
ReadVariableOp_4x
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack|
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_1|
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_2?
strided_slice_5StridedSliceReadVariableOp_4:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_5?
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_1?
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource_0^ReadVariableOp_4*
_output_shapes
:*
dtype02
ReadVariableOp_5x
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack|
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_6/stack_1|
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_5:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
strided_slice_6?
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_2?
ReadVariableOp_6ReadVariableOpreadvariableop_6_resource_0*
_output_shapes

:*
dtype02
ReadVariableOp_6
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSliceReadVariableOp_6:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_7y
MatMul_3MatMulplaceholder_2strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2

MatMul_3?
ReadVariableOp_7ReadVariableOpreadvariableop_6_resource_0^ReadVariableOp_6*
_output_shapes

:*
dtype02
ReadVariableOp_7
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack?
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack_1?
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2?
strided_slice_8StridedSliceReadVariableOp_7:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_8y
MatMul_4MatMulplaceholder_2strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1\
MulMuladd:z:0Const:output:0*
T0*'
_output_shapes
:?????????2
Mulb
Add_1AddMul:z:0Const_1:output:0*
T0*'
_output_shapes
:?????????2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_valueq
add_2AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3d
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*'
_output_shapes
:?????????2
Mul_1d
Add_3Add	Mul_1:z:0Const_3:output:0*
T0*'
_output_shapes
:?????????2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1k
mul_2Mulclip_by_value_1:z:0placeholder_2*
T0*'
_output_shapes
:?????????2
mul_2?
ReadVariableOp_8ReadVariableOpreadvariableop_6_resource_0^ReadVariableOp_7*
_output_shapes

:*
dtype02
ReadVariableOp_8
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stack?
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_9/stack_1?
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack_2?
strided_slice_9StridedSliceReadVariableOp_8:value:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_9u
MatMul_5MatMul	mul_2:z:0strided_slice_9:output:0*
T0*'
_output_shapes
:?????????2

MatMul_5q
add_4AddV2BiasAdd_2:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
add_4Q
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:?????????2
Tanhi
mul_3Mulclip_by_value:z:0placeholder_2*
T0*'
_output_shapes
:?????????2
mul_3S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xf
subSubsub/x:output:0clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
subZ
mul_4Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_4_
add_5AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:?????????2
add_5?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_5:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_6/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_6/yW
add_6AddV2placeholderadd_6/y:output:0*
T0*
_output_shapes
: 2
add_6T
add_7/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_7/y^
add_7AddV2while_loop_counteradd_7/y:output:0*
T0*
_output_shapes
: 2
add_7?
IdentityIdentity	add_7:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: 2

Identity?

Identity_1Identitywhile_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: 2

Identity_1?

Identity_2Identity	add_6:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: 2

Identity_3?

Identity_4Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*'
_output_shapes
:?????????2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"8
readvariableop_3_resourcereadvariableop_3_resource_0"8
readvariableop_6_resourcereadvariableop_6_resource_0"4
readvariableop_resourcereadvariableop_resource_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????: : :::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_8
?v
?
while_body_678704
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
readvariableop_resource_0
readvariableop_3_resource_0
readvariableop_6_resource_0
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
readvariableop_resource
readvariableop_3_resource
readvariableop_6_resource??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?ReadVariableOp_4?ReadVariableOp_5?ReadVariableOp_6?ReadVariableOp_7?ReadVariableOp_8?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemz
ReadVariableOpReadVariableOpreadvariableop_resource_0*
_output_shapes

:*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice?
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice:output:0*
T0*'
_output_shapes
:?????????2
MatMul?
ReadVariableOp_1ReadVariableOpreadvariableop_resource_0^ReadVariableOp*
_output_shapes

:*
dtype02
ReadVariableOp_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_1:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_2?
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2

MatMul_1?
ReadVariableOp_2ReadVariableOpreadvariableop_resource_0^ReadVariableOp_1*
_output_shapes

:*
dtype02
ReadVariableOp_2
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_2:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_3?
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2

MatMul_2|
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource_0*
_output_shapes
:*
dtype02
ReadVariableOp_3x
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_4/stack|
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2?
strided_slice_4StridedSliceReadVariableOp_3:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slice_4{
BiasAddBiasAddMatMul:product:0strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource_0^ReadVariableOp_3*
_output_shapes
:*
dtype02
ReadVariableOp_4x
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack|
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_1|
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_2?
strided_slice_5StridedSliceReadVariableOp_4:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_5?
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_1?
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource_0^ReadVariableOp_4*
_output_shapes
:*
dtype02
ReadVariableOp_5x
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack|
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_6/stack_1|
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_5:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
strided_slice_6?
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_2?
ReadVariableOp_6ReadVariableOpreadvariableop_6_resource_0*
_output_shapes

:*
dtype02
ReadVariableOp_6
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSliceReadVariableOp_6:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_7y
MatMul_3MatMulplaceholder_2strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2

MatMul_3?
ReadVariableOp_7ReadVariableOpreadvariableop_6_resource_0^ReadVariableOp_6*
_output_shapes

:*
dtype02
ReadVariableOp_7
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack?
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack_1?
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2?
strided_slice_8StridedSliceReadVariableOp_7:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_8y
MatMul_4MatMulplaceholder_2strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1\
MulMuladd:z:0Const:output:0*
T0*'
_output_shapes
:?????????2
Mulb
Add_1AddMul:z:0Const_1:output:0*
T0*'
_output_shapes
:?????????2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_valueq
add_2AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3d
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*'
_output_shapes
:?????????2
Mul_1d
Add_3Add	Mul_1:z:0Const_3:output:0*
T0*'
_output_shapes
:?????????2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1k
mul_2Mulclip_by_value_1:z:0placeholder_2*
T0*'
_output_shapes
:?????????2
mul_2?
ReadVariableOp_8ReadVariableOpreadvariableop_6_resource_0^ReadVariableOp_7*
_output_shapes

:*
dtype02
ReadVariableOp_8
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stack?
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_9/stack_1?
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack_2?
strided_slice_9StridedSliceReadVariableOp_8:value:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_9u
MatMul_5MatMul	mul_2:z:0strided_slice_9:output:0*
T0*'
_output_shapes
:?????????2

MatMul_5q
add_4AddV2BiasAdd_2:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
add_4Q
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:?????????2
Tanhi
mul_3Mulclip_by_value:z:0placeholder_2*
T0*'
_output_shapes
:?????????2
mul_3S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xf
subSubsub/x:output:0clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
subZ
mul_4Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_4_
add_5AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:?????????2
add_5?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_5:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_6/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_6/yW
add_6AddV2placeholderadd_6/y:output:0*
T0*
_output_shapes
: 2
add_6T
add_7/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_7/y^
add_7AddV2while_loop_counteradd_7/y:output:0*
T0*
_output_shapes
: 2
add_7?
IdentityIdentity	add_7:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: 2

Identity?

Identity_1Identitywhile_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: 2

Identity_1?

Identity_2Identity	add_6:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: 2

Identity_3?

Identity_4Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*'
_output_shapes
:?????????2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"8
readvariableop_3_resourcereadvariableop_3_resource_0"8
readvariableop_6_resourcereadvariableop_6_resource_0"4
readvariableop_resourcereadvariableop_resource_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????: : :::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_8
?v
?
while_body_680799
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
readvariableop_resource_0
readvariableop_3_resource_0
readvariableop_6_resource_0
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
readvariableop_resource
readvariableop_3_resource
readvariableop_6_resource??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?ReadVariableOp_4?ReadVariableOp_5?ReadVariableOp_6?ReadVariableOp_7?ReadVariableOp_8?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemz
ReadVariableOpReadVariableOpreadvariableop_resource_0*
_output_shapes

:*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice?
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice:output:0*
T0*'
_output_shapes
:?????????2
MatMul?
ReadVariableOp_1ReadVariableOpreadvariableop_resource_0^ReadVariableOp*
_output_shapes

:*
dtype02
ReadVariableOp_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_1:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_2?
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2

MatMul_1?
ReadVariableOp_2ReadVariableOpreadvariableop_resource_0^ReadVariableOp_1*
_output_shapes

:*
dtype02
ReadVariableOp_2
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_2:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_3?
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2

MatMul_2|
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource_0*
_output_shapes
:*
dtype02
ReadVariableOp_3x
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_4/stack|
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2?
strided_slice_4StridedSliceReadVariableOp_3:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slice_4{
BiasAddBiasAddMatMul:product:0strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource_0^ReadVariableOp_3*
_output_shapes
:*
dtype02
ReadVariableOp_4x
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack|
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_1|
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_2?
strided_slice_5StridedSliceReadVariableOp_4:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_5?
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_1?
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource_0^ReadVariableOp_4*
_output_shapes
:*
dtype02
ReadVariableOp_5x
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack|
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_6/stack_1|
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_5:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
strided_slice_6?
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_2?
ReadVariableOp_6ReadVariableOpreadvariableop_6_resource_0*
_output_shapes

:*
dtype02
ReadVariableOp_6
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSliceReadVariableOp_6:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_7y
MatMul_3MatMulplaceholder_2strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2

MatMul_3?
ReadVariableOp_7ReadVariableOpreadvariableop_6_resource_0^ReadVariableOp_6*
_output_shapes

:*
dtype02
ReadVariableOp_7
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack?
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack_1?
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2?
strided_slice_8StridedSliceReadVariableOp_7:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_8y
MatMul_4MatMulplaceholder_2strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1\
MulMuladd:z:0Const:output:0*
T0*'
_output_shapes
:?????????2
Mulb
Add_1AddMul:z:0Const_1:output:0*
T0*'
_output_shapes
:?????????2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_valueq
add_2AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3d
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*'
_output_shapes
:?????????2
Mul_1d
Add_3Add	Mul_1:z:0Const_3:output:0*
T0*'
_output_shapes
:?????????2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1k
mul_2Mulclip_by_value_1:z:0placeholder_2*
T0*'
_output_shapes
:?????????2
mul_2?
ReadVariableOp_8ReadVariableOpreadvariableop_6_resource_0^ReadVariableOp_7*
_output_shapes

:*
dtype02
ReadVariableOp_8
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stack?
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_9/stack_1?
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack_2?
strided_slice_9StridedSliceReadVariableOp_8:value:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_9u
MatMul_5MatMul	mul_2:z:0strided_slice_9:output:0*
T0*'
_output_shapes
:?????????2

MatMul_5q
add_4AddV2BiasAdd_2:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
add_4Q
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:?????????2
Tanhi
mul_3Mulclip_by_value:z:0placeholder_2*
T0*'
_output_shapes
:?????????2
mul_3S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xf
subSubsub/x:output:0clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
subZ
mul_4Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_4_
add_5AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:?????????2
add_5?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_5:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_6/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_6/yW
add_6AddV2placeholderadd_6/y:output:0*
T0*
_output_shapes
: 2
add_6T
add_7/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_7/y^
add_7AddV2while_loop_counteradd_7/y:output:0*
T0*
_output_shapes
: 2
add_7?
IdentityIdentity	add_7:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: 2

Identity?

Identity_1Identitywhile_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: 2

Identity_1?

Identity_2Identity	add_6:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: 2

Identity_3?

Identity_4Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*'
_output_shapes
:?????????2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"8
readvariableop_3_resourcereadvariableop_3_resource_0"8
readvariableop_6_resourcereadvariableop_6_resource_0"4
readvariableop_resourcereadvariableop_resource_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????: : :::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_8
?v
?
while_body_678960
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
readvariableop_resource_0
readvariableop_3_resource_0
readvariableop_6_resource_0
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
readvariableop_resource
readvariableop_3_resource
readvariableop_6_resource??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?ReadVariableOp_4?ReadVariableOp_5?ReadVariableOp_6?ReadVariableOp_7?ReadVariableOp_8?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemz
ReadVariableOpReadVariableOpreadvariableop_resource_0*
_output_shapes

:*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice?
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice:output:0*
T0*'
_output_shapes
:?????????2
MatMul?
ReadVariableOp_1ReadVariableOpreadvariableop_resource_0^ReadVariableOp*
_output_shapes

:*
dtype02
ReadVariableOp_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_1:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_2?
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2

MatMul_1?
ReadVariableOp_2ReadVariableOpreadvariableop_resource_0^ReadVariableOp_1*
_output_shapes

:*
dtype02
ReadVariableOp_2
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_2:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_3?
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2

MatMul_2|
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource_0*
_output_shapes
:*
dtype02
ReadVariableOp_3x
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_4/stack|
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2?
strided_slice_4StridedSliceReadVariableOp_3:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slice_4{
BiasAddBiasAddMatMul:product:0strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource_0^ReadVariableOp_3*
_output_shapes
:*
dtype02
ReadVariableOp_4x
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack|
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_1|
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_2?
strided_slice_5StridedSliceReadVariableOp_4:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_5?
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_1?
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource_0^ReadVariableOp_4*
_output_shapes
:*
dtype02
ReadVariableOp_5x
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack|
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_6/stack_1|
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_5:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
strided_slice_6?
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_2?
ReadVariableOp_6ReadVariableOpreadvariableop_6_resource_0*
_output_shapes

:*
dtype02
ReadVariableOp_6
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSliceReadVariableOp_6:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_7y
MatMul_3MatMulplaceholder_2strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2

MatMul_3?
ReadVariableOp_7ReadVariableOpreadvariableop_6_resource_0^ReadVariableOp_6*
_output_shapes

:*
dtype02
ReadVariableOp_7
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack?
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack_1?
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2?
strided_slice_8StridedSliceReadVariableOp_7:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_8y
MatMul_4MatMulplaceholder_2strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1\
MulMuladd:z:0Const:output:0*
T0*'
_output_shapes
:?????????2
Mulb
Add_1AddMul:z:0Const_1:output:0*
T0*'
_output_shapes
:?????????2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_valueq
add_2AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3d
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*'
_output_shapes
:?????????2
Mul_1d
Add_3Add	Mul_1:z:0Const_3:output:0*
T0*'
_output_shapes
:?????????2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1k
mul_2Mulclip_by_value_1:z:0placeholder_2*
T0*'
_output_shapes
:?????????2
mul_2?
ReadVariableOp_8ReadVariableOpreadvariableop_6_resource_0^ReadVariableOp_7*
_output_shapes

:*
dtype02
ReadVariableOp_8
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stack?
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_9/stack_1?
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack_2?
strided_slice_9StridedSliceReadVariableOp_8:value:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_9u
MatMul_5MatMul	mul_2:z:0strided_slice_9:output:0*
T0*'
_output_shapes
:?????????2

MatMul_5q
add_4AddV2BiasAdd_2:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
add_4Q
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:?????????2
Tanhi
mul_3Mulclip_by_value:z:0placeholder_2*
T0*'
_output_shapes
:?????????2
mul_3S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xf
subSubsub/x:output:0clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
subZ
mul_4Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_4_
add_5AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:?????????2
add_5?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_5:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_6/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_6/yW
add_6AddV2placeholderadd_6/y:output:0*
T0*
_output_shapes
: 2
add_6T
add_7/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_7/y^
add_7AddV2while_loop_counteradd_7/y:output:0*
T0*
_output_shapes
: 2
add_7?
IdentityIdentity	add_7:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: 2

Identity?

Identity_1Identitywhile_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: 2

Identity_1?

Identity_2Identity	add_6:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: 2

Identity_3?

Identity_4Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*'
_output_shapes
:?????????2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"8
readvariableop_3_resourcereadvariableop_3_resource_0"8
readvariableop_6_resourcereadvariableop_6_resource_0"4
readvariableop_resourcereadvariableop_resource_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????: : :::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_8
?
?
while_cond_678959
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1.
*while_cond_678959___redundant_placeholder0.
*while_cond_678959___redundant_placeholder1.
*while_cond_678959___redundant_placeholder2.
*while_cond_678959___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-: : : : :?????????: ::::
?
X
,__inference_concatenate_layer_call_fn_680966
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????(**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_6791262
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:????????? :( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?
?
gru_while_cond_679708
gru_while_loop_counter 
gru_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_gru_strided_slice_12
.gru_while_cond_679708___redundant_placeholder02
.gru_while_cond_679708___redundant_placeholder12
.gru_while_cond_679708___redundant_placeholder22
.gru_while_cond_679708___redundant_placeholder3
identity
\
LessLessplaceholderless_gru_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-: : : : :?????????: ::::
?
?
while_body_678408
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 statefulpartitionedcall_args_2_0$
 statefulpartitionedcall_args_3_0$
 statefulpartitionedcall_args_4_0
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4??StatefulPartitionedCall?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2 statefulpartitionedcall_args_2_0 statefulpartitionedcall_args_3_0 statefulpartitionedcall_args_4_0*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*:
_output_shapes(
&:?????????:?????????**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_gru_cell_layer_call_and_return_conditional_losses_6780832
StatefulPartitionedCall?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1f
IdentityIdentity	add_1:z:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identityy

Identity_1Identitywhile_maximum_iterations^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1h

Identity_2Identityadd:z:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_3?

Identity_4Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"B
statefulpartitionedcall_args_2 statefulpartitionedcall_args_2_0"B
statefulpartitionedcall_args_3 statefulpartitionedcall_args_3_0"B
statefulpartitionedcall_args_4 statefulpartitionedcall_args_4_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????: : :::22
StatefulPartitionedCallStatefulPartitionedCall
?a
?
D__inference_gru_cell_layer_call_and_return_conditional_losses_681117

inputs
states_0
readvariableop_resource
readvariableop_3_resource
readvariableop_6_resource
identity

identity_1??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?ReadVariableOp_4?ReadVariableOp_5?ReadVariableOp_6?ReadVariableOp_7?ReadVariableOp_8x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slicel
MatMulMatMulinputsstrided_slice:output:0*
T0*'
_output_shapes
:?????????2
MatMul?
ReadVariableOp_1ReadVariableOpreadvariableop_resource^ReadVariableOp*
_output_shapes

:*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_1r
MatMul_1MatMulinputsstrided_slice_1:output:0*
T0*'
_output_shapes
:?????????2

MatMul_1?
ReadVariableOp_2ReadVariableOpreadvariableop_resource^ReadVariableOp_1*
_output_shapes

:*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_2r
MatMul_2MatMulinputsstrided_slice_2:output:0*
T0*'
_output_shapes
:?????????2

MatMul_2z
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype02
ReadVariableOp_3x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slice_3{
BiasAddBiasAddMatMul:product:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource^ReadVariableOp_3*
_output_shapes
:*
dtype02
ReadVariableOp_4x
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack|
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2?
strided_slice_4StridedSliceReadVariableOp_4:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_4?
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_1?
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource^ReadVariableOp_4*
_output_shapes
:*
dtype02
ReadVariableOp_5x
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack|
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_5/stack_1|
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_2?
strided_slice_5StridedSliceReadVariableOp_5:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
strided_slice_5?
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_2~
ReadVariableOp_6ReadVariableOpreadvariableop_6_resource*
_output_shapes

:*
dtype02
ReadVariableOp_6
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_6:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_6t
MatMul_3MatMulstates_0strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2

MatMul_3?
ReadVariableOp_7ReadVariableOpreadvariableop_6_resource^ReadVariableOp_6*
_output_shapes

:*
dtype02
ReadVariableOp_7
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSliceReadVariableOp_7:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_7t
MatMul_4MatMulstates_0strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1\
MulMuladd:z:0Const:output:0*
T0*'
_output_shapes
:?????????2
Mulb
Add_1AddMul:z:0Const_1:output:0*
T0*'
_output_shapes
:?????????2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_valueq
add_2AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3d
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*'
_output_shapes
:?????????2
Mul_1d
Add_3Add	Mul_1:z:0Const_3:output:0*
T0*'
_output_shapes
:?????????2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1f
mul_2Mulclip_by_value_1:z:0states_0*
T0*'
_output_shapes
:?????????2
mul_2?
ReadVariableOp_8ReadVariableOpreadvariableop_6_resource^ReadVariableOp_7*
_output_shapes

:*
dtype02
ReadVariableOp_8
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack?
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_8/stack_1?
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2?
strided_slice_8StridedSliceReadVariableOp_8:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_8u
MatMul_5MatMul	mul_2:z:0strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2

MatMul_5q
add_4AddV2BiasAdd_2:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
add_4Q
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:?????????2
Tanhd
mul_3Mulclip_by_value:z:0states_0*
T0*'
_output_shapes
:?????????2
mul_3S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xf
subSubsub/x:output:0clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
subZ
mul_4Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_4_
add_5AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:?????????2
add_5?
IdentityIdentity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:?????????:?????????:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_8:& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0
?

?
&__inference_model_layer_call_fn_679897
inputs_0
inputs_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_6792852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:????????? :?????????:::::::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?
?
while_cond_678703
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1.
*while_cond_678703___redundant_placeholder0.
*while_cond_678703___redundant_placeholder1.
*while_cond_678703___redundant_placeholder2.
*while_cond_678703___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-: : : : :?????????: ::::
??
?
A__inference_model_layer_call_and_return_conditional_losses_679590
inputs_0
inputs_1
gru_readvariableop_resource!
gru_readvariableop_3_resource!
gru_readvariableop_6_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?gru/ReadVariableOp?gru/ReadVariableOp_1?gru/ReadVariableOp_2?gru/ReadVariableOp_3?gru/ReadVariableOp_4?gru/ReadVariableOp_5?gru/ReadVariableOp_6?gru/ReadVariableOp_7?gru/ReadVariableOp_8?	gru/whileN
	gru/ShapeShapeinputs_1*
T0*
_output_shapes
:2
	gru/Shape|
gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice/stack?
gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_1?
gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_2?
gru/strided_sliceStridedSlicegru/Shape:output:0 gru/strided_slice/stack:output:0"gru/strided_slice/stack_1:output:0"gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/strided_sliced
gru/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/zeros/mul/y|
gru/zeros/mulMulgru/strided_slice:output:0gru/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru/zeros/mulg
gru/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
gru/zeros/Less/yw
gru/zeros/LessLessgru/zeros/mul:z:0gru/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru/zeros/Lessj
gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
gru/zeros/packed/1?
gru/zeros/packedPackgru/strided_slice:output:0gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru/zeros/packedg
gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/zeros/Const?
	gru/zerosFillgru/zeros/packed:output:0gru/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
	gru/zeros}
gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose/perm?
gru/transpose	Transposeinputs_1gru/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
gru/transpose[
gru/Shape_1Shapegru/transpose:y:0*
T0*
_output_shapes
:2
gru/Shape_1?
gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_1/stack?
gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_1/stack_1?
gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_1/stack_2?
gru/strided_slice_1StridedSlicegru/Shape_1:output:0"gru/strided_slice_1/stack:output:0$gru/strided_slice_1/stack_1:output:0$gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/strided_slice_1?
gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
gru/TensorArrayV2/element_shape?
gru/TensorArrayV2TensorListReserve(gru/TensorArrayV2/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/TensorArrayV2?
9gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2;
9gru/TensorArrayUnstack/TensorListFromTensor/element_shape?
+gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru/transpose:y:0Bgru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+gru/TensorArrayUnstack/TensorListFromTensor?
gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_2/stack?
gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_2/stack_1?
gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_2/stack_2?
gru/strided_slice_2StridedSlicegru/transpose:y:0"gru/strided_slice_2/stack:output:0$gru/strided_slice_2/stack_1:output:0$gru/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
gru/strided_slice_2?
gru/ReadVariableOpReadVariableOpgru_readvariableop_resource*
_output_shapes

:*
dtype02
gru/ReadVariableOp?
gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
gru/strided_slice_3/stack?
gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
gru/strided_slice_3/stack_1?
gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
gru/strided_slice_3/stack_2?
gru/strided_slice_3StridedSlicegru/ReadVariableOp:value:0"gru/strided_slice_3/stack:output:0$gru/strided_slice_3/stack_1:output:0$gru/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru/strided_slice_3?

gru/MatMulMatMulgru/strided_slice_2:output:0gru/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2

gru/MatMul?
gru/ReadVariableOp_1ReadVariableOpgru_readvariableop_resource^gru/ReadVariableOp*
_output_shapes

:*
dtype02
gru/ReadVariableOp_1?
gru/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
gru/strided_slice_4/stack?
gru/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
gru/strided_slice_4/stack_1?
gru/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
gru/strided_slice_4/stack_2?
gru/strided_slice_4StridedSlicegru/ReadVariableOp_1:value:0"gru/strided_slice_4/stack:output:0$gru/strided_slice_4/stack_1:output:0$gru/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru/strided_slice_4?
gru/MatMul_1MatMulgru/strided_slice_2:output:0gru/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
gru/MatMul_1?
gru/ReadVariableOp_2ReadVariableOpgru_readvariableop_resource^gru/ReadVariableOp_1*
_output_shapes

:*
dtype02
gru/ReadVariableOp_2?
gru/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
gru/strided_slice_5/stack?
gru/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
gru/strided_slice_5/stack_1?
gru/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
gru/strided_slice_5/stack_2?
gru/strided_slice_5StridedSlicegru/ReadVariableOp_2:value:0"gru/strided_slice_5/stack:output:0$gru/strided_slice_5/stack_1:output:0$gru/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru/strided_slice_5?
gru/MatMul_2MatMulgru/strided_slice_2:output:0gru/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
gru/MatMul_2?
gru/ReadVariableOp_3ReadVariableOpgru_readvariableop_3_resource*
_output_shapes
:*
dtype02
gru/ReadVariableOp_3?
gru/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_6/stack?
gru/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_6/stack_1?
gru/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_6/stack_2?
gru/strided_slice_6StridedSlicegru/ReadVariableOp_3:value:0"gru/strided_slice_6/stack:output:0$gru/strided_slice_6/stack_1:output:0$gru/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
gru/strided_slice_6?
gru/BiasAddBiasAddgru/MatMul:product:0gru/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2
gru/BiasAdd?
gru/ReadVariableOp_4ReadVariableOpgru_readvariableop_3_resource^gru/ReadVariableOp_3*
_output_shapes
:*
dtype02
gru/ReadVariableOp_4?
gru/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_7/stack?
gru/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_7/stack_1?
gru/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_7/stack_2?
gru/strided_slice_7StridedSlicegru/ReadVariableOp_4:value:0"gru/strided_slice_7/stack:output:0$gru/strided_slice_7/stack_1:output:0$gru/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
gru/strided_slice_7?
gru/BiasAdd_1BiasAddgru/MatMul_1:product:0gru/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2
gru/BiasAdd_1?
gru/ReadVariableOp_5ReadVariableOpgru_readvariableop_3_resource^gru/ReadVariableOp_4*
_output_shapes
:*
dtype02
gru/ReadVariableOp_5?
gru/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_8/stack?
gru/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_8/stack_1?
gru/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_8/stack_2?
gru/strided_slice_8StridedSlicegru/ReadVariableOp_5:value:0"gru/strided_slice_8/stack:output:0$gru/strided_slice_8/stack_1:output:0$gru/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
gru/strided_slice_8?
gru/BiasAdd_2BiasAddgru/MatMul_2:product:0gru/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2
gru/BiasAdd_2?
gru/ReadVariableOp_6ReadVariableOpgru_readvariableop_6_resource*
_output_shapes

:*
dtype02
gru/ReadVariableOp_6?
gru/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
gru/strided_slice_9/stack?
gru/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
gru/strided_slice_9/stack_1?
gru/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
gru/strided_slice_9/stack_2?
gru/strided_slice_9StridedSlicegru/ReadVariableOp_6:value:0"gru/strided_slice_9/stack:output:0$gru/strided_slice_9/stack_1:output:0$gru/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru/strided_slice_9?
gru/MatMul_3MatMulgru/zeros:output:0gru/strided_slice_9:output:0*
T0*'
_output_shapes
:?????????2
gru/MatMul_3?
gru/ReadVariableOp_7ReadVariableOpgru_readvariableop_6_resource^gru/ReadVariableOp_6*
_output_shapes

:*
dtype02
gru/ReadVariableOp_7?
gru/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
gru/strided_slice_10/stack?
gru/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
gru/strided_slice_10/stack_1?
gru/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
gru/strided_slice_10/stack_2?
gru/strided_slice_10StridedSlicegru/ReadVariableOp_7:value:0#gru/strided_slice_10/stack:output:0%gru/strided_slice_10/stack_1:output:0%gru/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru/strided_slice_10?
gru/MatMul_4MatMulgru/zeros:output:0gru/strided_slice_10:output:0*
T0*'
_output_shapes
:?????????2
gru/MatMul_4{
gru/addAddV2gru/BiasAdd:output:0gru/MatMul_3:product:0*
T0*'
_output_shapes
:?????????2	
gru/add[
	gru/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
	gru/Const_
gru/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gru/Const_1l
gru/MulMulgru/add:z:0gru/Const:output:0*
T0*'
_output_shapes
:?????????2	
gru/Mulr
	gru/Add_1Addgru/Mul:z:0gru/Const_1:output:0*
T0*'
_output_shapes
:?????????2
	gru/Add_1
gru/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru/clip_by_value/Minimum/y?
gru/clip_by_value/MinimumMinimumgru/Add_1:z:0$gru/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
gru/clip_by_value/Minimumo
gru/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/clip_by_value/y?
gru/clip_by_valueMaximumgru/clip_by_value/Minimum:z:0gru/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
gru/clip_by_value?
	gru/add_2AddV2gru/BiasAdd_1:output:0gru/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
	gru/add_2_
gru/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
gru/Const_2_
gru/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gru/Const_3t
	gru/Mul_1Mulgru/add_2:z:0gru/Const_2:output:0*
T0*'
_output_shapes
:?????????2
	gru/Mul_1t
	gru/Add_3Addgru/Mul_1:z:0gru/Const_3:output:0*
T0*'
_output_shapes
:?????????2
	gru/Add_3?
gru/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru/clip_by_value_1/Minimum/y?
gru/clip_by_value_1/MinimumMinimumgru/Add_3:z:0&gru/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
gru/clip_by_value_1/Minimums
gru/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/clip_by_value_1/y?
gru/clip_by_value_1Maximumgru/clip_by_value_1/Minimum:z:0gru/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2
gru/clip_by_value_1|
	gru/mul_2Mulgru/clip_by_value_1:z:0gru/zeros:output:0*
T0*'
_output_shapes
:?????????2
	gru/mul_2?
gru/ReadVariableOp_8ReadVariableOpgru_readvariableop_6_resource^gru/ReadVariableOp_7*
_output_shapes

:*
dtype02
gru/ReadVariableOp_8?
gru/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
gru/strided_slice_11/stack?
gru/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
gru/strided_slice_11/stack_1?
gru/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
gru/strided_slice_11/stack_2?
gru/strided_slice_11StridedSlicegru/ReadVariableOp_8:value:0#gru/strided_slice_11/stack:output:0%gru/strided_slice_11/stack_1:output:0%gru/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru/strided_slice_11?
gru/MatMul_5MatMulgru/mul_2:z:0gru/strided_slice_11:output:0*
T0*'
_output_shapes
:?????????2
gru/MatMul_5?
	gru/add_4AddV2gru/BiasAdd_2:output:0gru/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
	gru/add_4]
gru/TanhTanhgru/add_4:z:0*
T0*'
_output_shapes
:?????????2

gru/Tanhz
	gru/mul_3Mulgru/clip_by_value:z:0gru/zeros:output:0*
T0*'
_output_shapes
:?????????2
	gru/mul_3[
	gru/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
	gru/sub/xv
gru/subSubgru/sub/x:output:0gru/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2	
gru/subj
	gru/mul_4Mulgru/sub:z:0gru/Tanh:y:0*
T0*'
_output_shapes
:?????????2
	gru/mul_4o
	gru/add_5AddV2gru/mul_3:z:0gru/mul_4:z:0*
T0*'
_output_shapes
:?????????2
	gru/add_5?
!gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2#
!gru/TensorArrayV2_1/element_shape?
gru/TensorArrayV2_1TensorListReserve*gru/TensorArrayV2_1/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/TensorArrayV2_1V
gru/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

gru/time?
gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru/while/maximum_iterationsr
gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru/while/loop_counter?
	gru/whileWhilegru/while/loop_counter:output:0%gru/while/maximum_iterations:output:0gru/time:output:0gru/TensorArrayV2_1:handle:0gru/zeros:output:0gru/strided_slice_1:output:0;gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_readvariableop_resourcegru_readvariableop_3_resourcegru_readvariableop_6_resource^gru/ReadVariableOp_2^gru/ReadVariableOp_5^gru/ReadVariableOp_8*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????: : : : : *!
bodyR
gru_while_body_679428*!
condR
gru_while_cond_679427*8
output_shapes'
%: : : : :?????????: : : : : *
parallel_iterations 2
	gru/while?
4gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   26
4gru/TensorArrayV2Stack/TensorListStack/element_shape?
&gru/TensorArrayV2Stack/TensorListStackTensorListStackgru/while:output:3=gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype02(
&gru/TensorArrayV2Stack/TensorListStack?
gru/strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
gru/strided_slice_12/stack?
gru/strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_12/stack_1?
gru/strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_12/stack_2?
gru/strided_slice_12StridedSlice/gru/TensorArrayV2Stack/TensorListStack:tensor:0#gru/strided_slice_12/stack:output:0%gru/strided_slice_12/stack_1:output:0%gru/strided_slice_12/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
gru/strided_slice_12?
gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose_1/perm?
gru/transpose_1	Transpose/gru/TensorArrayV2Stack/TensorListStack:tensor:0gru/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????2
gru/transpose_1t
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2gru/strided_slice_12:output:0inputs_0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????(2
concatenate/concat?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:((*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulconcatenate/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
dense/BiasAddj

re_lu/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2

re_lu/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulre_lu/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddf
reshape/ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapedense_1/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
reshape/Reshape?
IdentityIdentityreshape/Reshape:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^gru/ReadVariableOp^gru/ReadVariableOp_1^gru/ReadVariableOp_2^gru/ReadVariableOp_3^gru/ReadVariableOp_4^gru/ReadVariableOp_5^gru/ReadVariableOp_6^gru/ReadVariableOp_7^gru/ReadVariableOp_8
^gru/while*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:????????? :?????????:::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2(
gru/ReadVariableOpgru/ReadVariableOp2,
gru/ReadVariableOp_1gru/ReadVariableOp_12,
gru/ReadVariableOp_2gru/ReadVariableOp_22,
gru/ReadVariableOp_3gru/ReadVariableOp_32,
gru/ReadVariableOp_4gru/ReadVariableOp_42,
gru/ReadVariableOp_5gru/ReadVariableOp_52,
gru/ReadVariableOp_6gru/ReadVariableOp_62,
gru/ReadVariableOp_7gru/ReadVariableOp_72,
gru/ReadVariableOp_8gru/ReadVariableOp_82
	gru/while	gru/while:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?
?
A__inference_model_layer_call_and_return_conditional_losses_679214
input_2
input_1&
"gru_statefulpartitionedcall_args_1&
"gru_statefulpartitionedcall_args_2&
"gru_statefulpartitionedcall_args_3(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?gru/StatefulPartitionedCall?
gru/StatefulPartitionedCallStatefulPartitionedCallinput_1"gru_statefulpartitionedcall_args_1"gru_statefulpartitionedcall_args_2"gru_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_6788422
gru/StatefulPartitionedCall?
concatenate/PartitionedCallPartitionedCall$gru/StatefulPartitionedCall:output:0input_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????(**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_6791262
concatenate/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????(**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_6791452
dense/StatefulPartitionedCall?
re_lu/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????(**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_6791622
re_lu/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_6791802!
dense_1/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_6792052
reshape/PartitionedCall?
IdentityIdentity reshape/PartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^gru/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:????????? :?????????:::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall:' #
!
_user_specified_name	input_2:'#
!
_user_specified_name	input_1
?

?
&__inference_model_layer_call_fn_679264
input_2
input_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2input_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_6792542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:????????? :?????????:::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_2:'#
!
_user_specified_name	input_1
?
?
__inference__traced_save_681274
file_prefix-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop/
+savev2_dense_1_1_kernel_read_readvariableop-
)savev2_dense_1_1_bias_read_readvariableop+
'savev2_gru_1_kernel_read_readvariableop5
1savev2_gru_1_recurrent_kernel_read_readvariableop)
%savev2_gru_1_bias_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_81331bf0115f487a886f07facc2fd4a8/part2
StringJoin/inputs_1?

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop+savev2_dense_1_1_kernel_read_readvariableop)savev2_dense_1_1_bias_read_readvariableop'savev2_gru_1_kernel_read_readvariableop1savev2_gru_1_recurrent_kernel_read_readvariableop%savev2_gru_1_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes
	22
SaveV2?
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1?
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names?
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity?

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*Q
_input_shapes@
>: :((:(:(::::: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
?
_
C__inference_reshape_layer_call_and_return_conditional_losses_681023

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?
?
while_cond_680270
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1.
*while_cond_680270___redundant_placeholder0.
*while_cond_680270___redundant_placeholder1.
*while_cond_680270___redundant_placeholder2.
*while_cond_680270___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-: : : : :?????????: ::::
?a
?
D__inference_gru_cell_layer_call_and_return_conditional_losses_681206

inputs
states_0
readvariableop_resource
readvariableop_3_resource
readvariableop_6_resource
identity

identity_1??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?ReadVariableOp_4?ReadVariableOp_5?ReadVariableOp_6?ReadVariableOp_7?ReadVariableOp_8x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slicel
MatMulMatMulinputsstrided_slice:output:0*
T0*'
_output_shapes
:?????????2
MatMul?
ReadVariableOp_1ReadVariableOpreadvariableop_resource^ReadVariableOp*
_output_shapes

:*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_1r
MatMul_1MatMulinputsstrided_slice_1:output:0*
T0*'
_output_shapes
:?????????2

MatMul_1?
ReadVariableOp_2ReadVariableOpreadvariableop_resource^ReadVariableOp_1*
_output_shapes

:*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_2r
MatMul_2MatMulinputsstrided_slice_2:output:0*
T0*'
_output_shapes
:?????????2

MatMul_2z
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype02
ReadVariableOp_3x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slice_3{
BiasAddBiasAddMatMul:product:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource^ReadVariableOp_3*
_output_shapes
:*
dtype02
ReadVariableOp_4x
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack|
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2?
strided_slice_4StridedSliceReadVariableOp_4:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_4?
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_1?
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource^ReadVariableOp_4*
_output_shapes
:*
dtype02
ReadVariableOp_5x
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack|
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_5/stack_1|
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_2?
strided_slice_5StridedSliceReadVariableOp_5:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
strided_slice_5?
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_2~
ReadVariableOp_6ReadVariableOpreadvariableop_6_resource*
_output_shapes

:*
dtype02
ReadVariableOp_6
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_6:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_6t
MatMul_3MatMulstates_0strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2

MatMul_3?
ReadVariableOp_7ReadVariableOpreadvariableop_6_resource^ReadVariableOp_6*
_output_shapes

:*
dtype02
ReadVariableOp_7
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSliceReadVariableOp_7:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_7t
MatMul_4MatMulstates_0strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1\
MulMuladd:z:0Const:output:0*
T0*'
_output_shapes
:?????????2
Mulb
Add_1AddMul:z:0Const_1:output:0*
T0*'
_output_shapes
:?????????2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_valueq
add_2AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3d
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*'
_output_shapes
:?????????2
Mul_1d
Add_3Add	Mul_1:z:0Const_3:output:0*
T0*'
_output_shapes
:?????????2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1f
mul_2Mulclip_by_value_1:z:0states_0*
T0*'
_output_shapes
:?????????2
mul_2?
ReadVariableOp_8ReadVariableOpreadvariableop_6_resource^ReadVariableOp_7*
_output_shapes

:*
dtype02
ReadVariableOp_8
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack?
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_8/stack_1?
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2?
strided_slice_8StridedSliceReadVariableOp_8:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_8u
MatMul_5MatMul	mul_2:z:0strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2

MatMul_5q
add_4AddV2BiasAdd_2:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
add_4Q
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:?????????2
Tanhd
mul_3Mulclip_by_value:z:0states_0*
T0*'
_output_shapes
:?????????2
mul_3S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xf
subSubsub/x:output:0clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
subZ
mul_4Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_4_
add_5AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:?????????2
add_5?
IdentityIdentity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:?????????:?????????:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_8:& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0
?w
?
model_gru_while_body_677800 
model_gru_while_loop_counter&
"model_gru_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
model_gru_strided_slice_1_0[
Wtensorarrayv2read_tensorlistgetitem_model_gru_tensorarrayunstack_tensorlistfromtensor_0
readvariableop_resource_0
readvariableop_3_resource_0
readvariableop_6_resource_0
identity

identity_1

identity_2

identity_3

identity_4
model_gru_strided_slice_1Y
Utensorarrayv2read_tensorlistgetitem_model_gru_tensorarrayunstack_tensorlistfromtensor
readvariableop_resource
readvariableop_3_resource
readvariableop_6_resource??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?ReadVariableOp_4?ReadVariableOp_5?ReadVariableOp_6?ReadVariableOp_7?ReadVariableOp_8?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemWtensorarrayv2read_tensorlistgetitem_model_gru_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemz
ReadVariableOpReadVariableOpreadvariableop_resource_0*
_output_shapes

:*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice?
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice:output:0*
T0*'
_output_shapes
:?????????2
MatMul?
ReadVariableOp_1ReadVariableOpreadvariableop_resource_0^ReadVariableOp*
_output_shapes

:*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_1?
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2

MatMul_1?
ReadVariableOp_2ReadVariableOpreadvariableop_resource_0^ReadVariableOp_1*
_output_shapes

:*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_2?
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2

MatMul_2|
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource_0*
_output_shapes
:*
dtype02
ReadVariableOp_3x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slice_3{
BiasAddBiasAddMatMul:product:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource_0^ReadVariableOp_3*
_output_shapes
:*
dtype02
ReadVariableOp_4x
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack|
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2?
strided_slice_4StridedSliceReadVariableOp_4:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_4?
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_1?
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource_0^ReadVariableOp_4*
_output_shapes
:*
dtype02
ReadVariableOp_5x
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack|
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_5/stack_1|
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_2?
strided_slice_5StridedSliceReadVariableOp_5:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
strided_slice_5?
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_2?
ReadVariableOp_6ReadVariableOpreadvariableop_6_resource_0*
_output_shapes

:*
dtype02
ReadVariableOp_6
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_6:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_6y
MatMul_3MatMulplaceholder_2strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2

MatMul_3?
ReadVariableOp_7ReadVariableOpreadvariableop_6_resource_0^ReadVariableOp_6*
_output_shapes

:*
dtype02
ReadVariableOp_7
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSliceReadVariableOp_7:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_7y
MatMul_4MatMulplaceholder_2strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1\
MulMuladd:z:0Const:output:0*
T0*'
_output_shapes
:?????????2
Mulb
Add_1AddMul:z:0Const_1:output:0*
T0*'
_output_shapes
:?????????2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_valueq
add_2AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3d
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*'
_output_shapes
:?????????2
Mul_1d
Add_3Add	Mul_1:z:0Const_3:output:0*
T0*'
_output_shapes
:?????????2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1k
mul_2Mulclip_by_value_1:z:0placeholder_2*
T0*'
_output_shapes
:?????????2
mul_2?
ReadVariableOp_8ReadVariableOpreadvariableop_6_resource_0^ReadVariableOp_7*
_output_shapes

:*
dtype02
ReadVariableOp_8
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack?
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_8/stack_1?
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2?
strided_slice_8StridedSliceReadVariableOp_8:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_8u
MatMul_5MatMul	mul_2:z:0strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2

MatMul_5q
add_4AddV2BiasAdd_2:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
add_4Q
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:?????????2
Tanhi
mul_3Mulclip_by_value:z:0placeholder_2*
T0*'
_output_shapes
:?????????2
mul_3S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xf
subSubsub/x:output:0clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
subZ
mul_4Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_4_
add_5AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:?????????2
add_5?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_5:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_6/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_6/yW
add_6AddV2placeholderadd_6/y:output:0*
T0*
_output_shapes
: 2
add_6T
add_7/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_7/yh
add_7AddV2model_gru_while_loop_counteradd_7/y:output:0*
T0*
_output_shapes
: 2
add_7?
IdentityIdentity	add_7:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: 2

Identity?

Identity_1Identity"model_gru_while_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: 2

Identity_1?

Identity_2Identity	add_6:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: 2

Identity_3?

Identity_4Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*'
_output_shapes
:?????????2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"8
model_gru_strided_slice_1model_gru_strided_slice_1_0"8
readvariableop_3_resourcereadvariableop_3_resource_0"8
readvariableop_6_resourcereadvariableop_6_resource_0"4
readvariableop_resourcereadvariableop_resource_0"?
Utensorarrayv2read_tensorlistgetitem_model_gru_tensorarrayunstack_tensorlistfromtensorWtensorarrayv2read_tensorlistgetitem_model_gru_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????: : :::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_8
?
?
while_cond_680542
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1.
*while_cond_680542___redundant_placeholder0.
*while_cond_680542___redundant_placeholder1.
*while_cond_680542___redundant_placeholder2.
*while_cond_680542___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-: : : : :?????????: ::::
?w
?
gru_while_body_679428
gru_while_loop_counter 
gru_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
gru_strided_slice_1_0U
Qtensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0
readvariableop_resource_0
readvariableop_3_resource_0
readvariableop_6_resource_0
identity

identity_1

identity_2

identity_3

identity_4
gru_strided_slice_1S
Otensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor
readvariableop_resource
readvariableop_3_resource
readvariableop_6_resource??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?ReadVariableOp_4?ReadVariableOp_5?ReadVariableOp_6?ReadVariableOp_7?ReadVariableOp_8?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemQtensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemz
ReadVariableOpReadVariableOpreadvariableop_resource_0*
_output_shapes

:*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice?
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice:output:0*
T0*'
_output_shapes
:?????????2
MatMul?
ReadVariableOp_1ReadVariableOpreadvariableop_resource_0^ReadVariableOp*
_output_shapes

:*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_1?
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2

MatMul_1?
ReadVariableOp_2ReadVariableOpreadvariableop_resource_0^ReadVariableOp_1*
_output_shapes

:*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_2?
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2

MatMul_2|
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource_0*
_output_shapes
:*
dtype02
ReadVariableOp_3x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slice_3{
BiasAddBiasAddMatMul:product:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource_0^ReadVariableOp_3*
_output_shapes
:*
dtype02
ReadVariableOp_4x
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack|
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2?
strided_slice_4StridedSliceReadVariableOp_4:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_4?
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_1?
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource_0^ReadVariableOp_4*
_output_shapes
:*
dtype02
ReadVariableOp_5x
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack|
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_5/stack_1|
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_2?
strided_slice_5StridedSliceReadVariableOp_5:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
strided_slice_5?
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_2?
ReadVariableOp_6ReadVariableOpreadvariableop_6_resource_0*
_output_shapes

:*
dtype02
ReadVariableOp_6
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_6:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_6y
MatMul_3MatMulplaceholder_2strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2

MatMul_3?
ReadVariableOp_7ReadVariableOpreadvariableop_6_resource_0^ReadVariableOp_6*
_output_shapes

:*
dtype02
ReadVariableOp_7
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSliceReadVariableOp_7:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_7y
MatMul_4MatMulplaceholder_2strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1\
MulMuladd:z:0Const:output:0*
T0*'
_output_shapes
:?????????2
Mulb
Add_1AddMul:z:0Const_1:output:0*
T0*'
_output_shapes
:?????????2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_valueq
add_2AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3d
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*'
_output_shapes
:?????????2
Mul_1d
Add_3Add	Mul_1:z:0Const_3:output:0*
T0*'
_output_shapes
:?????????2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1k
mul_2Mulclip_by_value_1:z:0placeholder_2*
T0*'
_output_shapes
:?????????2
mul_2?
ReadVariableOp_8ReadVariableOpreadvariableop_6_resource_0^ReadVariableOp_7*
_output_shapes

:*
dtype02
ReadVariableOp_8
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack?
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_8/stack_1?
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2?
strided_slice_8StridedSliceReadVariableOp_8:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_8u
MatMul_5MatMul	mul_2:z:0strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2

MatMul_5q
add_4AddV2BiasAdd_2:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
add_4Q
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:?????????2
Tanhi
mul_3Mulclip_by_value:z:0placeholder_2*
T0*'
_output_shapes
:?????????2
mul_3S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xf
subSubsub/x:output:0clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
subZ
mul_4Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_4_
add_5AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:?????????2
add_5?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_5:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_6/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_6/yW
add_6AddV2placeholderadd_6/y:output:0*
T0*
_output_shapes
: 2
add_6T
add_7/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_7/yb
add_7AddV2gru_while_loop_counteradd_7/y:output:0*
T0*
_output_shapes
: 2
add_7?
IdentityIdentity	add_7:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: 2

Identity?

Identity_1Identitygru_while_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: 2

Identity_1?

Identity_2Identity	add_6:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: 2

Identity_3?

Identity_4Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*'
_output_shapes
:?????????2

Identity_4",
gru_strided_slice_1gru_strided_slice_1_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"8
readvariableop_3_resourcereadvariableop_3_resource_0"8
readvariableop_6_resourcereadvariableop_6_resource_0"4
readvariableop_resourcereadvariableop_resource_0"?
Otensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensorQtensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????: : :::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_8
ے
?
?__inference_gru_layer_call_and_return_conditional_losses_680937

inputs
readvariableop_resource
readvariableop_3_resource
readvariableop_6_resource
identity??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?ReadVariableOp_4?ReadVariableOp_5?ReadVariableOp_6?ReadVariableOp_7?ReadVariableOp_8?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype02
ReadVariableOp
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_3?
MatMulMatMulstrided_slice_2:output:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
MatMul?
ReadVariableOp_1ReadVariableOpreadvariableop_resource^ReadVariableOp*
_output_shapes

:*
dtype02
ReadVariableOp_1
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSliceReadVariableOp_1:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_4?
MatMul_1MatMulstrided_slice_2:output:0strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2

MatMul_1?
ReadVariableOp_2ReadVariableOpreadvariableop_resource^ReadVariableOp_1*
_output_shapes

:*
dtype02
ReadVariableOp_2
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack?
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_5/stack_1?
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2?
strided_slice_5StridedSliceReadVariableOp_2:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_5?
MatMul_2MatMulstrided_slice_2:output:0strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2

MatMul_2z
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype02
ReadVariableOp_3x
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_6/stack|
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack_1|
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_3:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slice_6{
BiasAddBiasAddMatMul:product:0strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource^ReadVariableOp_3*
_output_shapes
:*
dtype02
ReadVariableOp_4x
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_7/stack|
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_7/stack_1|
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_7/stack_2?
strided_slice_7StridedSliceReadVariableOp_4:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_7?
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_1?
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource^ReadVariableOp_4*
_output_shapes
:*
dtype02
ReadVariableOp_5x
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack|
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_8/stack_1|
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack_2?
strided_slice_8StridedSliceReadVariableOp_5:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
strided_slice_8?
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_2~
ReadVariableOp_6ReadVariableOpreadvariableop_6_resource*
_output_shapes

:*
dtype02
ReadVariableOp_6
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_9/stack?
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stack_1?
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack_2?
strided_slice_9StridedSliceReadVariableOp_6:value:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_9z
MatMul_3MatMulzeros:output:0strided_slice_9:output:0*
T0*'
_output_shapes
:?????????2

MatMul_3?
ReadVariableOp_7ReadVariableOpreadvariableop_6_resource^ReadVariableOp_6*
_output_shapes

:*
dtype02
ReadVariableOp_7?
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_10/stack?
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_10/stack_1?
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_10/stack_2?
strided_slice_10StridedSliceReadVariableOp_7:value:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_10{
MatMul_4MatMulzeros:output:0strided_slice_10:output:0*
T0*'
_output_shapes
:?????????2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1\
MulMuladd:z:0Const:output:0*
T0*'
_output_shapes
:?????????2
Mulb
Add_1AddMul:z:0Const_1:output:0*
T0*'
_output_shapes
:?????????2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_valueq
add_2AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3d
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*'
_output_shapes
:?????????2
Mul_1d
Add_3Add	Mul_1:z:0Const_3:output:0*
T0*'
_output_shapes
:?????????2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1l
mul_2Mulclip_by_value_1:z:0zeros:output:0*
T0*'
_output_shapes
:?????????2
mul_2?
ReadVariableOp_8ReadVariableOpreadvariableop_6_resource^ReadVariableOp_7*
_output_shapes

:*
dtype02
ReadVariableOp_8?
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_11/stack?
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_11/stack_1?
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_11/stack_2?
strided_slice_11StridedSliceReadVariableOp_8:value:0strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_11v
MatMul_5MatMul	mul_2:z:0strided_slice_11:output:0*
T0*'
_output_shapes
:?????????2

MatMul_5q
add_4AddV2BiasAdd_2:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
add_4Q
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:?????????2
Tanhj
mul_3Mulclip_by_value:z:0zeros:output:0*
T0*'
_output_shapes
:?????????2
mul_3S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xf
subSubsub/x:output:0clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
subZ
mul_4Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_4_
add_5AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:?????????2
add_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0readvariableop_resourcereadvariableop_3_resourcereadvariableop_6_resource^ReadVariableOp_2^ReadVariableOp_5^ReadVariableOp_8*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????: : : : : *
bodyR
while_body_680799*
condR
while_cond_680798*8
output_shapes'
%: : : : :?????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_12/stack~
strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_12/stack_1~
strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_12/stack_2?
strided_slice_12StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_12y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????2
transpose_1?
IdentityIdentitystrided_slice_12:output:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8^while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_82
whilewhile:& "
 
_user_specified_nameinputs
?a
?
D__inference_gru_cell_layer_call_and_return_conditional_losses_678083

inputs

states
readvariableop_resource
readvariableop_3_resource
readvariableop_6_resource
identity

identity_1??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?ReadVariableOp_4?ReadVariableOp_5?ReadVariableOp_6?ReadVariableOp_7?ReadVariableOp_8x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slicel
MatMulMatMulinputsstrided_slice:output:0*
T0*'
_output_shapes
:?????????2
MatMul?
ReadVariableOp_1ReadVariableOpreadvariableop_resource^ReadVariableOp*
_output_shapes

:*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_1r
MatMul_1MatMulinputsstrided_slice_1:output:0*
T0*'
_output_shapes
:?????????2

MatMul_1?
ReadVariableOp_2ReadVariableOpreadvariableop_resource^ReadVariableOp_1*
_output_shapes

:*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_2r
MatMul_2MatMulinputsstrided_slice_2:output:0*
T0*'
_output_shapes
:?????????2

MatMul_2z
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype02
ReadVariableOp_3x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slice_3{
BiasAddBiasAddMatMul:product:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource^ReadVariableOp_3*
_output_shapes
:*
dtype02
ReadVariableOp_4x
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack|
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2?
strided_slice_4StridedSliceReadVariableOp_4:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_4?
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_1?
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource^ReadVariableOp_4*
_output_shapes
:*
dtype02
ReadVariableOp_5x
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack|
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_5/stack_1|
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_2?
strided_slice_5StridedSliceReadVariableOp_5:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
strided_slice_5?
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_2~
ReadVariableOp_6ReadVariableOpreadvariableop_6_resource*
_output_shapes

:*
dtype02
ReadVariableOp_6
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_6:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_6r
MatMul_3MatMulstatesstrided_slice_6:output:0*
T0*'
_output_shapes
:?????????2

MatMul_3?
ReadVariableOp_7ReadVariableOpreadvariableop_6_resource^ReadVariableOp_6*
_output_shapes

:*
dtype02
ReadVariableOp_7
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSliceReadVariableOp_7:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_7r
MatMul_4MatMulstatesstrided_slice_7:output:0*
T0*'
_output_shapes
:?????????2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1\
MulMuladd:z:0Const:output:0*
T0*'
_output_shapes
:?????????2
Mulb
Add_1AddMul:z:0Const_1:output:0*
T0*'
_output_shapes
:?????????2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_valueq
add_2AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3d
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*'
_output_shapes
:?????????2
Mul_1d
Add_3Add	Mul_1:z:0Const_3:output:0*
T0*'
_output_shapes
:?????????2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1d
mul_2Mulclip_by_value_1:z:0states*
T0*'
_output_shapes
:?????????2
mul_2?
ReadVariableOp_8ReadVariableOpreadvariableop_6_resource^ReadVariableOp_7*
_output_shapes

:*
dtype02
ReadVariableOp_8
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack?
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_8/stack_1?
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2?
strided_slice_8StridedSliceReadVariableOp_8:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_8u
MatMul_5MatMul	mul_2:z:0strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2

MatMul_5q
add_4AddV2BiasAdd_2:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
add_4Q
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:?????????2
Tanhb
mul_3Mulclip_by_value:z:0states*
T0*'
_output_shapes
:?????????2
mul_3S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xf
subSubsub/x:output:0clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
subZ
mul_4Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_4_
add_5AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:?????????2
add_5?
IdentityIdentity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:?????????:?????????:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_8:& "
 
_user_specified_nameinputs:&"
 
_user_specified_namestates
?w
?
gru_while_body_679709
gru_while_loop_counter 
gru_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
gru_strided_slice_1_0U
Qtensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0
readvariableop_resource_0
readvariableop_3_resource_0
readvariableop_6_resource_0
identity

identity_1

identity_2

identity_3

identity_4
gru_strided_slice_1S
Otensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor
readvariableop_resource
readvariableop_3_resource
readvariableop_6_resource??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?ReadVariableOp_4?ReadVariableOp_5?ReadVariableOp_6?ReadVariableOp_7?ReadVariableOp_8?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemQtensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemz
ReadVariableOpReadVariableOpreadvariableop_resource_0*
_output_shapes

:*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice?
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice:output:0*
T0*'
_output_shapes
:?????????2
MatMul?
ReadVariableOp_1ReadVariableOpreadvariableop_resource_0^ReadVariableOp*
_output_shapes

:*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_1?
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice_1:output:0*
T0*'
_output_shapes
:?????????2

MatMul_1?
ReadVariableOp_2ReadVariableOpreadvariableop_resource_0^ReadVariableOp_1*
_output_shapes

:*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_2?
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2

MatMul_2|
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource_0*
_output_shapes
:*
dtype02
ReadVariableOp_3x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slice_3{
BiasAddBiasAddMatMul:product:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource_0^ReadVariableOp_3*
_output_shapes
:*
dtype02
ReadVariableOp_4x
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack|
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2?
strided_slice_4StridedSliceReadVariableOp_4:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_4?
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_1?
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource_0^ReadVariableOp_4*
_output_shapes
:*
dtype02
ReadVariableOp_5x
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack|
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_5/stack_1|
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_2?
strided_slice_5StridedSliceReadVariableOp_5:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
strided_slice_5?
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_2?
ReadVariableOp_6ReadVariableOpreadvariableop_6_resource_0*
_output_shapes

:*
dtype02
ReadVariableOp_6
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_6:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_6y
MatMul_3MatMulplaceholder_2strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2

MatMul_3?
ReadVariableOp_7ReadVariableOpreadvariableop_6_resource_0^ReadVariableOp_6*
_output_shapes

:*
dtype02
ReadVariableOp_7
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSliceReadVariableOp_7:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_7y
MatMul_4MatMulplaceholder_2strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1\
MulMuladd:z:0Const:output:0*
T0*'
_output_shapes
:?????????2
Mulb
Add_1AddMul:z:0Const_1:output:0*
T0*'
_output_shapes
:?????????2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_valueq
add_2AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3d
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*'
_output_shapes
:?????????2
Mul_1d
Add_3Add	Mul_1:z:0Const_3:output:0*
T0*'
_output_shapes
:?????????2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1k
mul_2Mulclip_by_value_1:z:0placeholder_2*
T0*'
_output_shapes
:?????????2
mul_2?
ReadVariableOp_8ReadVariableOpreadvariableop_6_resource_0^ReadVariableOp_7*
_output_shapes

:*
dtype02
ReadVariableOp_8
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack?
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_8/stack_1?
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2?
strided_slice_8StridedSliceReadVariableOp_8:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_8u
MatMul_5MatMul	mul_2:z:0strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2

MatMul_5q
add_4AddV2BiasAdd_2:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
add_4Q
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:?????????2
Tanhi
mul_3Mulclip_by_value:z:0placeholder_2*
T0*'
_output_shapes
:?????????2
mul_3S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xf
subSubsub/x:output:0clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
subZ
mul_4Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_4_
add_5AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:?????????2
add_5?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_5:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_6/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_6/yW
add_6AddV2placeholderadd_6/y:output:0*
T0*
_output_shapes
: 2
add_6T
add_7/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_7/yb
add_7AddV2gru_while_loop_counteradd_7/y:output:0*
T0*
_output_shapes
: 2
add_7?
IdentityIdentity	add_7:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: 2

Identity?

Identity_1Identitygru_while_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: 2

Identity_1?

Identity_2Identity	add_6:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: 2

Identity_3?

Identity_4Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*'
_output_shapes
:?????????2

Identity_4",
gru_strided_slice_1gru_strided_slice_1_0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"8
readvariableop_3_resourcereadvariableop_3_resource_0"8
readvariableop_6_resourcereadvariableop_6_resource_0"4
readvariableop_resourcereadvariableop_resource_0"?
Otensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensorQtensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????: : :::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_8
?v
?
while_body_680271
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
readvariableop_resource_0
readvariableop_3_resource_0
readvariableop_6_resource_0
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
readvariableop_resource
readvariableop_3_resource
readvariableop_6_resource??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?ReadVariableOp_4?ReadVariableOp_5?ReadVariableOp_6?ReadVariableOp_7?ReadVariableOp_8?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemz
ReadVariableOpReadVariableOpreadvariableop_resource_0*
_output_shapes

:*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice?
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice:output:0*
T0*'
_output_shapes
:?????????2
MatMul?
ReadVariableOp_1ReadVariableOpreadvariableop_resource_0^ReadVariableOp*
_output_shapes

:*
dtype02
ReadVariableOp_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_1:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_2?
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2

MatMul_1?
ReadVariableOp_2ReadVariableOpreadvariableop_resource_0^ReadVariableOp_1*
_output_shapes

:*
dtype02
ReadVariableOp_2
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_2:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_3?
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2

MatMul_2|
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource_0*
_output_shapes
:*
dtype02
ReadVariableOp_3x
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_4/stack|
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2?
strided_slice_4StridedSliceReadVariableOp_3:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slice_4{
BiasAddBiasAddMatMul:product:0strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource_0^ReadVariableOp_3*
_output_shapes
:*
dtype02
ReadVariableOp_4x
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack|
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_1|
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_2?
strided_slice_5StridedSliceReadVariableOp_4:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_5?
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_1?
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource_0^ReadVariableOp_4*
_output_shapes
:*
dtype02
ReadVariableOp_5x
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack|
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_6/stack_1|
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_5:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
strided_slice_6?
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_2?
ReadVariableOp_6ReadVariableOpreadvariableop_6_resource_0*
_output_shapes

:*
dtype02
ReadVariableOp_6
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSliceReadVariableOp_6:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_7y
MatMul_3MatMulplaceholder_2strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2

MatMul_3?
ReadVariableOp_7ReadVariableOpreadvariableop_6_resource_0^ReadVariableOp_6*
_output_shapes

:*
dtype02
ReadVariableOp_7
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack?
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack_1?
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2?
strided_slice_8StridedSliceReadVariableOp_7:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_8y
MatMul_4MatMulplaceholder_2strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1\
MulMuladd:z:0Const:output:0*
T0*'
_output_shapes
:?????????2
Mulb
Add_1AddMul:z:0Const_1:output:0*
T0*'
_output_shapes
:?????????2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_valueq
add_2AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3d
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*'
_output_shapes
:?????????2
Mul_1d
Add_3Add	Mul_1:z:0Const_3:output:0*
T0*'
_output_shapes
:?????????2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1k
mul_2Mulclip_by_value_1:z:0placeholder_2*
T0*'
_output_shapes
:?????????2
mul_2?
ReadVariableOp_8ReadVariableOpreadvariableop_6_resource_0^ReadVariableOp_7*
_output_shapes

:*
dtype02
ReadVariableOp_8
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stack?
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_9/stack_1?
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack_2?
strided_slice_9StridedSliceReadVariableOp_8:value:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_9u
MatMul_5MatMul	mul_2:z:0strided_slice_9:output:0*
T0*'
_output_shapes
:?????????2

MatMul_5q
add_4AddV2BiasAdd_2:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
add_4Q
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:?????????2
Tanhi
mul_3Mulclip_by_value:z:0placeholder_2*
T0*'
_output_shapes
:?????????2
mul_3S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xf
subSubsub/x:output:0clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
subZ
mul_4Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_4_
add_5AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:?????????2
add_5?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_5:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_6/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_6/yW
add_6AddV2placeholderadd_6/y:output:0*
T0*
_output_shapes
: 2
add_6T
add_7/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_7/y^
add_7AddV2while_loop_counteradd_7/y:output:0*
T0*
_output_shapes
: 2
add_7?
IdentityIdentity	add_7:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: 2

Identity?

Identity_1Identitywhile_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: 2

Identity_1?

Identity_2Identity	add_6:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: 2

Identity_3?

Identity_4Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*'
_output_shapes
:?????????2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"8
readvariableop_3_resourcereadvariableop_3_resource_0"8
readvariableop_6_resourcereadvariableop_6_resource_0"4
readvariableop_resourcereadvariableop_resource_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????: : :::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_8
?
?
while_body_678516
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 statefulpartitionedcall_args_2_0$
 statefulpartitionedcall_args_3_0$
 statefulpartitionedcall_args_4_0
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4??StatefulPartitionedCall?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02%
#TensorArrayV2Read/TensorListGetItem?
StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2 statefulpartitionedcall_args_2_0 statefulpartitionedcall_args_3_0 statefulpartitionedcall_args_4_0*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*:
_output_shapes(
&:?????????:?????????**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_gru_cell_layer_call_and_return_conditional_losses_6781722
StatefulPartitionedCall?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yQ
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: 2
addT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/y^
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: 2
add_1f
IdentityIdentity	add_1:z:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identityy

Identity_1Identitywhile_maximum_iterations^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1h

Identity_2Identityadd:z:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_3?

Identity_4Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"B
statefulpartitionedcall_args_2 statefulpartitionedcall_args_2_0"B
statefulpartitionedcall_args_3 statefulpartitionedcall_args_3_0"B
statefulpartitionedcall_args_4 statefulpartitionedcall_args_4_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????: : :::22
StatefulPartitionedCallStatefulPartitionedCall
?
?
gru_while_cond_679427
gru_while_loop_counter 
gru_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_gru_strided_slice_12
.gru_while_cond_679427___redundant_placeholder02
.gru_while_cond_679427___redundant_placeholder12
.gru_while_cond_679427___redundant_placeholder22
.gru_while_cond_679427___redundant_placeholder3
identity
\
LessLessplaceholderless_gru_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-: : : : :?????????: ::::
?
B
&__inference_re_lu_layer_call_fn_680993

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????(**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_6791622
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????(:& "
 
_user_specified_nameinputs
ے
?
?__inference_gru_layer_call_and_return_conditional_losses_678842

inputs
readvariableop_resource
readvariableop_3_resource
readvariableop_6_resource
identity??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?ReadVariableOp_4?ReadVariableOp_5?ReadVariableOp_6?ReadVariableOp_7?ReadVariableOp_8?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype02
ReadVariableOp
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_3?
MatMulMatMulstrided_slice_2:output:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
MatMul?
ReadVariableOp_1ReadVariableOpreadvariableop_resource^ReadVariableOp*
_output_shapes

:*
dtype02
ReadVariableOp_1
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSliceReadVariableOp_1:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_4?
MatMul_1MatMulstrided_slice_2:output:0strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2

MatMul_1?
ReadVariableOp_2ReadVariableOpreadvariableop_resource^ReadVariableOp_1*
_output_shapes

:*
dtype02
ReadVariableOp_2
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack?
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_5/stack_1?
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2?
strided_slice_5StridedSliceReadVariableOp_2:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_5?
MatMul_2MatMulstrided_slice_2:output:0strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2

MatMul_2z
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype02
ReadVariableOp_3x
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_6/stack|
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack_1|
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_3:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slice_6{
BiasAddBiasAddMatMul:product:0strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource^ReadVariableOp_3*
_output_shapes
:*
dtype02
ReadVariableOp_4x
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_7/stack|
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_7/stack_1|
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_7/stack_2?
strided_slice_7StridedSliceReadVariableOp_4:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_7?
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_1?
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource^ReadVariableOp_4*
_output_shapes
:*
dtype02
ReadVariableOp_5x
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack|
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_8/stack_1|
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack_2?
strided_slice_8StridedSliceReadVariableOp_5:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
strided_slice_8?
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_2~
ReadVariableOp_6ReadVariableOpreadvariableop_6_resource*
_output_shapes

:*
dtype02
ReadVariableOp_6
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_9/stack?
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stack_1?
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack_2?
strided_slice_9StridedSliceReadVariableOp_6:value:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_9z
MatMul_3MatMulzeros:output:0strided_slice_9:output:0*
T0*'
_output_shapes
:?????????2

MatMul_3?
ReadVariableOp_7ReadVariableOpreadvariableop_6_resource^ReadVariableOp_6*
_output_shapes

:*
dtype02
ReadVariableOp_7?
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_10/stack?
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_10/stack_1?
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_10/stack_2?
strided_slice_10StridedSliceReadVariableOp_7:value:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_10{
MatMul_4MatMulzeros:output:0strided_slice_10:output:0*
T0*'
_output_shapes
:?????????2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1\
MulMuladd:z:0Const:output:0*
T0*'
_output_shapes
:?????????2
Mulb
Add_1AddMul:z:0Const_1:output:0*
T0*'
_output_shapes
:?????????2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_valueq
add_2AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3d
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*'
_output_shapes
:?????????2
Mul_1d
Add_3Add	Mul_1:z:0Const_3:output:0*
T0*'
_output_shapes
:?????????2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1l
mul_2Mulclip_by_value_1:z:0zeros:output:0*
T0*'
_output_shapes
:?????????2
mul_2?
ReadVariableOp_8ReadVariableOpreadvariableop_6_resource^ReadVariableOp_7*
_output_shapes

:*
dtype02
ReadVariableOp_8?
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_11/stack?
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_11/stack_1?
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_11/stack_2?
strided_slice_11StridedSliceReadVariableOp_8:value:0strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_11v
MatMul_5MatMul	mul_2:z:0strided_slice_11:output:0*
T0*'
_output_shapes
:?????????2

MatMul_5q
add_4AddV2BiasAdd_2:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
add_4Q
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:?????????2
Tanhj
mul_3Mulclip_by_value:z:0zeros:output:0*
T0*'
_output_shapes
:?????????2
mul_3S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xf
subSubsub/x:output:0clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
subZ
mul_4Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_4_
add_5AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:?????????2
add_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0readvariableop_resourcereadvariableop_3_resourcereadvariableop_6_resource^ReadVariableOp_2^ReadVariableOp_5^ReadVariableOp_8*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????: : : : : *
bodyR
while_body_678704*
condR
while_cond_678703*8
output_shapes'
%: : : : :?????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_12/stack~
strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_12/stack_1~
strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_12/stack_2?
strided_slice_12StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_12y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????2
transpose_1?
IdentityIdentitystrided_slice_12:output:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8^while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_82
whilewhile:& "
 
_user_specified_nameinputs
?
?
A__inference_dense_layer_call_and_return_conditional_losses_680976

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:((*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????(::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
ے
?
?__inference_gru_layer_call_and_return_conditional_losses_680681

inputs
readvariableop_resource
readvariableop_3_resource
readvariableop_6_resource
identity??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?ReadVariableOp_4?ReadVariableOp_5?ReadVariableOp_6?ReadVariableOp_7?ReadVariableOp_8?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype02
ReadVariableOp
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_3?
MatMulMatMulstrided_slice_2:output:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
MatMul?
ReadVariableOp_1ReadVariableOpreadvariableop_resource^ReadVariableOp*
_output_shapes

:*
dtype02
ReadVariableOp_1
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSliceReadVariableOp_1:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_4?
MatMul_1MatMulstrided_slice_2:output:0strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2

MatMul_1?
ReadVariableOp_2ReadVariableOpreadvariableop_resource^ReadVariableOp_1*
_output_shapes

:*
dtype02
ReadVariableOp_2
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack?
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_5/stack_1?
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2?
strided_slice_5StridedSliceReadVariableOp_2:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_5?
MatMul_2MatMulstrided_slice_2:output:0strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2

MatMul_2z
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype02
ReadVariableOp_3x
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_6/stack|
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack_1|
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_3:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slice_6{
BiasAddBiasAddMatMul:product:0strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource^ReadVariableOp_3*
_output_shapes
:*
dtype02
ReadVariableOp_4x
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_7/stack|
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_7/stack_1|
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_7/stack_2?
strided_slice_7StridedSliceReadVariableOp_4:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_7?
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_1?
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource^ReadVariableOp_4*
_output_shapes
:*
dtype02
ReadVariableOp_5x
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack|
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_8/stack_1|
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack_2?
strided_slice_8StridedSliceReadVariableOp_5:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
strided_slice_8?
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_2~
ReadVariableOp_6ReadVariableOpreadvariableop_6_resource*
_output_shapes

:*
dtype02
ReadVariableOp_6
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_9/stack?
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stack_1?
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack_2?
strided_slice_9StridedSliceReadVariableOp_6:value:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_9z
MatMul_3MatMulzeros:output:0strided_slice_9:output:0*
T0*'
_output_shapes
:?????????2

MatMul_3?
ReadVariableOp_7ReadVariableOpreadvariableop_6_resource^ReadVariableOp_6*
_output_shapes

:*
dtype02
ReadVariableOp_7?
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_10/stack?
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_10/stack_1?
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_10/stack_2?
strided_slice_10StridedSliceReadVariableOp_7:value:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_10{
MatMul_4MatMulzeros:output:0strided_slice_10:output:0*
T0*'
_output_shapes
:?????????2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1\
MulMuladd:z:0Const:output:0*
T0*'
_output_shapes
:?????????2
Mulb
Add_1AddMul:z:0Const_1:output:0*
T0*'
_output_shapes
:?????????2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_valueq
add_2AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3d
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*'
_output_shapes
:?????????2
Mul_1d
Add_3Add	Mul_1:z:0Const_3:output:0*
T0*'
_output_shapes
:?????????2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1l
mul_2Mulclip_by_value_1:z:0zeros:output:0*
T0*'
_output_shapes
:?????????2
mul_2?
ReadVariableOp_8ReadVariableOpreadvariableop_6_resource^ReadVariableOp_7*
_output_shapes

:*
dtype02
ReadVariableOp_8?
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_11/stack?
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_11/stack_1?
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_11/stack_2?
strided_slice_11StridedSliceReadVariableOp_8:value:0strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_11v
MatMul_5MatMul	mul_2:z:0strided_slice_11:output:0*
T0*'
_output_shapes
:?????????2

MatMul_5q
add_4AddV2BiasAdd_2:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
add_4Q
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:?????????2
Tanhj
mul_3Mulclip_by_value:z:0zeros:output:0*
T0*'
_output_shapes
:?????????2
mul_3S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xf
subSubsub/x:output:0clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
subZ
mul_4Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_4_
add_5AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:?????????2
add_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0readvariableop_resourcereadvariableop_3_resourcereadvariableop_6_resource^ReadVariableOp_2^ReadVariableOp_5^ReadVariableOp_8*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????: : : : : *
bodyR
while_body_680543*
condR
while_cond_680542*8
output_shapes'
%: : : : :?????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_12/stack~
strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_12/stack_1~
strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_12/stack_2?
strided_slice_12StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_12y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????2
transpose_1?
IdentityIdentitystrided_slice_12:output:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8^while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_82
whilewhile:& "
 
_user_specified_nameinputs
??
?
?__inference_gru_layer_call_and_return_conditional_losses_680153
inputs_0
readvariableop_resource
readvariableop_3_resource
readvariableop_6_resource
identity??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?ReadVariableOp_4?ReadVariableOp_5?ReadVariableOp_6?ReadVariableOp_7?ReadVariableOp_8?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype02
ReadVariableOp
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_3?
MatMulMatMulstrided_slice_2:output:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
MatMul?
ReadVariableOp_1ReadVariableOpreadvariableop_resource^ReadVariableOp*
_output_shapes

:*
dtype02
ReadVariableOp_1
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSliceReadVariableOp_1:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_4?
MatMul_1MatMulstrided_slice_2:output:0strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2

MatMul_1?
ReadVariableOp_2ReadVariableOpreadvariableop_resource^ReadVariableOp_1*
_output_shapes

:*
dtype02
ReadVariableOp_2
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack?
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_5/stack_1?
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2?
strided_slice_5StridedSliceReadVariableOp_2:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_5?
MatMul_2MatMulstrided_slice_2:output:0strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2

MatMul_2z
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype02
ReadVariableOp_3x
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_6/stack|
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack_1|
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_3:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slice_6{
BiasAddBiasAddMatMul:product:0strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource^ReadVariableOp_3*
_output_shapes
:*
dtype02
ReadVariableOp_4x
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_7/stack|
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_7/stack_1|
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_7/stack_2?
strided_slice_7StridedSliceReadVariableOp_4:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_7?
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_1?
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource^ReadVariableOp_4*
_output_shapes
:*
dtype02
ReadVariableOp_5x
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack|
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_8/stack_1|
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack_2?
strided_slice_8StridedSliceReadVariableOp_5:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
strided_slice_8?
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_2~
ReadVariableOp_6ReadVariableOpreadvariableop_6_resource*
_output_shapes

:*
dtype02
ReadVariableOp_6
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_9/stack?
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stack_1?
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack_2?
strided_slice_9StridedSliceReadVariableOp_6:value:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_9z
MatMul_3MatMulzeros:output:0strided_slice_9:output:0*
T0*'
_output_shapes
:?????????2

MatMul_3?
ReadVariableOp_7ReadVariableOpreadvariableop_6_resource^ReadVariableOp_6*
_output_shapes

:*
dtype02
ReadVariableOp_7?
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_10/stack?
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_10/stack_1?
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_10/stack_2?
strided_slice_10StridedSliceReadVariableOp_7:value:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_10{
MatMul_4MatMulzeros:output:0strided_slice_10:output:0*
T0*'
_output_shapes
:?????????2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1\
MulMuladd:z:0Const:output:0*
T0*'
_output_shapes
:?????????2
Mulb
Add_1AddMul:z:0Const_1:output:0*
T0*'
_output_shapes
:?????????2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_valueq
add_2AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3d
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*'
_output_shapes
:?????????2
Mul_1d
Add_3Add	Mul_1:z:0Const_3:output:0*
T0*'
_output_shapes
:?????????2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1l
mul_2Mulclip_by_value_1:z:0zeros:output:0*
T0*'
_output_shapes
:?????????2
mul_2?
ReadVariableOp_8ReadVariableOpreadvariableop_6_resource^ReadVariableOp_7*
_output_shapes

:*
dtype02
ReadVariableOp_8?
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_11/stack?
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_11/stack_1?
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_11/stack_2?
strided_slice_11StridedSliceReadVariableOp_8:value:0strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_11v
MatMul_5MatMul	mul_2:z:0strided_slice_11:output:0*
T0*'
_output_shapes
:?????????2

MatMul_5q
add_4AddV2BiasAdd_2:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
add_4Q
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:?????????2
Tanhj
mul_3Mulclip_by_value:z:0zeros:output:0*
T0*'
_output_shapes
:?????????2
mul_3S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xf
subSubsub/x:output:0clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
subZ
mul_4Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_4_
add_5AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:?????????2
add_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0readvariableop_resourcereadvariableop_3_resourcereadvariableop_6_resource^ReadVariableOp_2^ReadVariableOp_5^ReadVariableOp_8*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????: : : : : *
bodyR
while_body_680015*
condR
while_cond_680014*8
output_shapes'
%: : : : :?????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_12/stack~
strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_12/stack_1~
strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_12/stack_2?
strided_slice_12StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_12y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_1?
IdentityIdentitystrided_slice_12:output:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8^while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_82
whilewhile:( $
"
_user_specified_name
inputs/0
?
?
C__inference_dense_1_layer_call_and_return_conditional_losses_681003

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????(::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?

?
&__inference_model_layer_call_fn_679884
inputs_0
inputs_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_6792542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:????????? :?????????:::::::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?;
?
?__inference_gru_layer_call_and_return_conditional_losses_678576

inputs"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*:
_output_shapes(
&:?????????:?????????**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_gru_cell_layer_call_and_return_conditional_losses_6781722
StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4^StatefulPartitionedCall*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????: : : : : *
bodyR
while_body_678516*
condR
while_cond_678515*8
output_shapes'
%: : : : :?????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_1?
IdentityIdentitystrided_slice_3:output:0^StatefulPartitionedCall^while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall2
whilewhile:& "
 
_user_specified_nameinputs
?
?
$__inference_gru_layer_call_fn_680425
inputs_0"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_6785762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0
?
?
(__inference_dense_1_layer_call_fn_681010

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_6791802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????(::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
model_gru_while_cond_677799 
model_gru_while_loop_counter&
"model_gru_while_maximum_iterations
placeholder
placeholder_1
placeholder_2"
less_model_gru_strided_slice_18
4model_gru_while_cond_677799___redundant_placeholder08
4model_gru_while_cond_677799___redundant_placeholder18
4model_gru_while_cond_677799___redundant_placeholder28
4model_gru_while_cond_677799___redundant_placeholder3
identity
b
LessLessplaceholderless_model_gru_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-: : : : :?????????: ::::
??
?
A__inference_model_layer_call_and_return_conditional_losses_679871
inputs_0
inputs_1
gru_readvariableop_resource!
gru_readvariableop_3_resource!
gru_readvariableop_6_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?gru/ReadVariableOp?gru/ReadVariableOp_1?gru/ReadVariableOp_2?gru/ReadVariableOp_3?gru/ReadVariableOp_4?gru/ReadVariableOp_5?gru/ReadVariableOp_6?gru/ReadVariableOp_7?gru/ReadVariableOp_8?	gru/whileN
	gru/ShapeShapeinputs_1*
T0*
_output_shapes
:2
	gru/Shape|
gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice/stack?
gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_1?
gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_2?
gru/strided_sliceStridedSlicegru/Shape:output:0 gru/strided_slice/stack:output:0"gru/strided_slice/stack_1:output:0"gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/strided_sliced
gru/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/zeros/mul/y|
gru/zeros/mulMulgru/strided_slice:output:0gru/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru/zeros/mulg
gru/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
gru/zeros/Less/yw
gru/zeros/LessLessgru/zeros/mul:z:0gru/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru/zeros/Lessj
gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
gru/zeros/packed/1?
gru/zeros/packedPackgru/strided_slice:output:0gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru/zeros/packedg
gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/zeros/Const?
	gru/zerosFillgru/zeros/packed:output:0gru/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
	gru/zeros}
gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose/perm?
gru/transpose	Transposeinputs_1gru/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
gru/transpose[
gru/Shape_1Shapegru/transpose:y:0*
T0*
_output_shapes
:2
gru/Shape_1?
gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_1/stack?
gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_1/stack_1?
gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_1/stack_2?
gru/strided_slice_1StridedSlicegru/Shape_1:output:0"gru/strided_slice_1/stack:output:0$gru/strided_slice_1/stack_1:output:0$gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/strided_slice_1?
gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
gru/TensorArrayV2/element_shape?
gru/TensorArrayV2TensorListReserve(gru/TensorArrayV2/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/TensorArrayV2?
9gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2;
9gru/TensorArrayUnstack/TensorListFromTensor/element_shape?
+gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru/transpose:y:0Bgru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+gru/TensorArrayUnstack/TensorListFromTensor?
gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_2/stack?
gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_2/stack_1?
gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_2/stack_2?
gru/strided_slice_2StridedSlicegru/transpose:y:0"gru/strided_slice_2/stack:output:0$gru/strided_slice_2/stack_1:output:0$gru/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
gru/strided_slice_2?
gru/ReadVariableOpReadVariableOpgru_readvariableop_resource*
_output_shapes

:*
dtype02
gru/ReadVariableOp?
gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
gru/strided_slice_3/stack?
gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
gru/strided_slice_3/stack_1?
gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
gru/strided_slice_3/stack_2?
gru/strided_slice_3StridedSlicegru/ReadVariableOp:value:0"gru/strided_slice_3/stack:output:0$gru/strided_slice_3/stack_1:output:0$gru/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru/strided_slice_3?

gru/MatMulMatMulgru/strided_slice_2:output:0gru/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2

gru/MatMul?
gru/ReadVariableOp_1ReadVariableOpgru_readvariableop_resource^gru/ReadVariableOp*
_output_shapes

:*
dtype02
gru/ReadVariableOp_1?
gru/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
gru/strided_slice_4/stack?
gru/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
gru/strided_slice_4/stack_1?
gru/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
gru/strided_slice_4/stack_2?
gru/strided_slice_4StridedSlicegru/ReadVariableOp_1:value:0"gru/strided_slice_4/stack:output:0$gru/strided_slice_4/stack_1:output:0$gru/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru/strided_slice_4?
gru/MatMul_1MatMulgru/strided_slice_2:output:0gru/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
gru/MatMul_1?
gru/ReadVariableOp_2ReadVariableOpgru_readvariableop_resource^gru/ReadVariableOp_1*
_output_shapes

:*
dtype02
gru/ReadVariableOp_2?
gru/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
gru/strided_slice_5/stack?
gru/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
gru/strided_slice_5/stack_1?
gru/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
gru/strided_slice_5/stack_2?
gru/strided_slice_5StridedSlicegru/ReadVariableOp_2:value:0"gru/strided_slice_5/stack:output:0$gru/strided_slice_5/stack_1:output:0$gru/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru/strided_slice_5?
gru/MatMul_2MatMulgru/strided_slice_2:output:0gru/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
gru/MatMul_2?
gru/ReadVariableOp_3ReadVariableOpgru_readvariableop_3_resource*
_output_shapes
:*
dtype02
gru/ReadVariableOp_3?
gru/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_6/stack?
gru/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_6/stack_1?
gru/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_6/stack_2?
gru/strided_slice_6StridedSlicegru/ReadVariableOp_3:value:0"gru/strided_slice_6/stack:output:0$gru/strided_slice_6/stack_1:output:0$gru/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
gru/strided_slice_6?
gru/BiasAddBiasAddgru/MatMul:product:0gru/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2
gru/BiasAdd?
gru/ReadVariableOp_4ReadVariableOpgru_readvariableop_3_resource^gru/ReadVariableOp_3*
_output_shapes
:*
dtype02
gru/ReadVariableOp_4?
gru/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_7/stack?
gru/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_7/stack_1?
gru/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_7/stack_2?
gru/strided_slice_7StridedSlicegru/ReadVariableOp_4:value:0"gru/strided_slice_7/stack:output:0$gru/strided_slice_7/stack_1:output:0$gru/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
gru/strided_slice_7?
gru/BiasAdd_1BiasAddgru/MatMul_1:product:0gru/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2
gru/BiasAdd_1?
gru/ReadVariableOp_5ReadVariableOpgru_readvariableop_3_resource^gru/ReadVariableOp_4*
_output_shapes
:*
dtype02
gru/ReadVariableOp_5?
gru/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_8/stack?
gru/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_8/stack_1?
gru/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_8/stack_2?
gru/strided_slice_8StridedSlicegru/ReadVariableOp_5:value:0"gru/strided_slice_8/stack:output:0$gru/strided_slice_8/stack_1:output:0$gru/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
gru/strided_slice_8?
gru/BiasAdd_2BiasAddgru/MatMul_2:product:0gru/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2
gru/BiasAdd_2?
gru/ReadVariableOp_6ReadVariableOpgru_readvariableop_6_resource*
_output_shapes

:*
dtype02
gru/ReadVariableOp_6?
gru/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
gru/strided_slice_9/stack?
gru/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
gru/strided_slice_9/stack_1?
gru/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
gru/strided_slice_9/stack_2?
gru/strided_slice_9StridedSlicegru/ReadVariableOp_6:value:0"gru/strided_slice_9/stack:output:0$gru/strided_slice_9/stack_1:output:0$gru/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru/strided_slice_9?
gru/MatMul_3MatMulgru/zeros:output:0gru/strided_slice_9:output:0*
T0*'
_output_shapes
:?????????2
gru/MatMul_3?
gru/ReadVariableOp_7ReadVariableOpgru_readvariableop_6_resource^gru/ReadVariableOp_6*
_output_shapes

:*
dtype02
gru/ReadVariableOp_7?
gru/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
gru/strided_slice_10/stack?
gru/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
gru/strided_slice_10/stack_1?
gru/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
gru/strided_slice_10/stack_2?
gru/strided_slice_10StridedSlicegru/ReadVariableOp_7:value:0#gru/strided_slice_10/stack:output:0%gru/strided_slice_10/stack_1:output:0%gru/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru/strided_slice_10?
gru/MatMul_4MatMulgru/zeros:output:0gru/strided_slice_10:output:0*
T0*'
_output_shapes
:?????????2
gru/MatMul_4{
gru/addAddV2gru/BiasAdd:output:0gru/MatMul_3:product:0*
T0*'
_output_shapes
:?????????2	
gru/add[
	gru/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
	gru/Const_
gru/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gru/Const_1l
gru/MulMulgru/add:z:0gru/Const:output:0*
T0*'
_output_shapes
:?????????2	
gru/Mulr
	gru/Add_1Addgru/Mul:z:0gru/Const_1:output:0*
T0*'
_output_shapes
:?????????2
	gru/Add_1
gru/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru/clip_by_value/Minimum/y?
gru/clip_by_value/MinimumMinimumgru/Add_1:z:0$gru/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
gru/clip_by_value/Minimumo
gru/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/clip_by_value/y?
gru/clip_by_valueMaximumgru/clip_by_value/Minimum:z:0gru/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
gru/clip_by_value?
	gru/add_2AddV2gru/BiasAdd_1:output:0gru/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
	gru/add_2_
gru/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
gru/Const_2_
gru/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gru/Const_3t
	gru/Mul_1Mulgru/add_2:z:0gru/Const_2:output:0*
T0*'
_output_shapes
:?????????2
	gru/Mul_1t
	gru/Add_3Addgru/Mul_1:z:0gru/Const_3:output:0*
T0*'
_output_shapes
:?????????2
	gru/Add_3?
gru/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru/clip_by_value_1/Minimum/y?
gru/clip_by_value_1/MinimumMinimumgru/Add_3:z:0&gru/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
gru/clip_by_value_1/Minimums
gru/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/clip_by_value_1/y?
gru/clip_by_value_1Maximumgru/clip_by_value_1/Minimum:z:0gru/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2
gru/clip_by_value_1|
	gru/mul_2Mulgru/clip_by_value_1:z:0gru/zeros:output:0*
T0*'
_output_shapes
:?????????2
	gru/mul_2?
gru/ReadVariableOp_8ReadVariableOpgru_readvariableop_6_resource^gru/ReadVariableOp_7*
_output_shapes

:*
dtype02
gru/ReadVariableOp_8?
gru/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
gru/strided_slice_11/stack?
gru/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
gru/strided_slice_11/stack_1?
gru/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
gru/strided_slice_11/stack_2?
gru/strided_slice_11StridedSlicegru/ReadVariableOp_8:value:0#gru/strided_slice_11/stack:output:0%gru/strided_slice_11/stack_1:output:0%gru/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
gru/strided_slice_11?
gru/MatMul_5MatMulgru/mul_2:z:0gru/strided_slice_11:output:0*
T0*'
_output_shapes
:?????????2
gru/MatMul_5?
	gru/add_4AddV2gru/BiasAdd_2:output:0gru/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
	gru/add_4]
gru/TanhTanhgru/add_4:z:0*
T0*'
_output_shapes
:?????????2

gru/Tanhz
	gru/mul_3Mulgru/clip_by_value:z:0gru/zeros:output:0*
T0*'
_output_shapes
:?????????2
	gru/mul_3[
	gru/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
	gru/sub/xv
gru/subSubgru/sub/x:output:0gru/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2	
gru/subj
	gru/mul_4Mulgru/sub:z:0gru/Tanh:y:0*
T0*'
_output_shapes
:?????????2
	gru/mul_4o
	gru/add_5AddV2gru/mul_3:z:0gru/mul_4:z:0*
T0*'
_output_shapes
:?????????2
	gru/add_5?
!gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2#
!gru/TensorArrayV2_1/element_shape?
gru/TensorArrayV2_1TensorListReserve*gru/TensorArrayV2_1/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/TensorArrayV2_1V
gru/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

gru/time?
gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru/while/maximum_iterationsr
gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru/while/loop_counter?
	gru/whileWhilegru/while/loop_counter:output:0%gru/while/maximum_iterations:output:0gru/time:output:0gru/TensorArrayV2_1:handle:0gru/zeros:output:0gru/strided_slice_1:output:0;gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_readvariableop_resourcegru_readvariableop_3_resourcegru_readvariableop_6_resource^gru/ReadVariableOp_2^gru/ReadVariableOp_5^gru/ReadVariableOp_8*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????: : : : : *!
bodyR
gru_while_body_679709*!
condR
gru_while_cond_679708*8
output_shapes'
%: : : : :?????????: : : : : *
parallel_iterations 2
	gru/while?
4gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   26
4gru/TensorArrayV2Stack/TensorListStack/element_shape?
&gru/TensorArrayV2Stack/TensorListStackTensorListStackgru/while:output:3=gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype02(
&gru/TensorArrayV2Stack/TensorListStack?
gru/strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
gru/strided_slice_12/stack?
gru/strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_12/stack_1?
gru/strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_12/stack_2?
gru/strided_slice_12StridedSlice/gru/TensorArrayV2Stack/TensorListStack:tensor:0#gru/strided_slice_12/stack:output:0%gru/strided_slice_12/stack_1:output:0%gru/strided_slice_12/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
gru/strided_slice_12?
gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose_1/perm?
gru/transpose_1	Transpose/gru/TensorArrayV2Stack/TensorListStack:tensor:0gru/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????2
gru/transpose_1t
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2gru/strided_slice_12:output:0inputs_0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????(2
concatenate/concat?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:((*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulconcatenate/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
dense/BiasAddj

re_lu/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2

re_lu/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulre_lu/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddf
reshape/ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapedense_1/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
reshape/Reshape?
IdentityIdentityreshape/Reshape:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^gru/ReadVariableOp^gru/ReadVariableOp_1^gru/ReadVariableOp_2^gru/ReadVariableOp_3^gru/ReadVariableOp_4^gru/ReadVariableOp_5^gru/ReadVariableOp_6^gru/ReadVariableOp_7^gru/ReadVariableOp_8
^gru/while*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:????????? :?????????:::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2(
gru/ReadVariableOpgru/ReadVariableOp2,
gru/ReadVariableOp_1gru/ReadVariableOp_12,
gru/ReadVariableOp_2gru/ReadVariableOp_22,
gru/ReadVariableOp_3gru/ReadVariableOp_32,
gru/ReadVariableOp_4gru/ReadVariableOp_42,
gru/ReadVariableOp_5gru/ReadVariableOp_52,
gru/ReadVariableOp_6gru/ReadVariableOp_62,
gru/ReadVariableOp_7gru/ReadVariableOp_72,
gru/ReadVariableOp_8gru/ReadVariableOp_82
	gru/while	gru/while:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
??
?
?__inference_gru_layer_call_and_return_conditional_losses_680409
inputs_0
readvariableop_resource
readvariableop_3_resource
readvariableop_6_resource
identity??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?ReadVariableOp_4?ReadVariableOp_5?ReadVariableOp_6?ReadVariableOp_7?ReadVariableOp_8?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype02
ReadVariableOp
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_3?
MatMulMatMulstrided_slice_2:output:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
MatMul?
ReadVariableOp_1ReadVariableOpreadvariableop_resource^ReadVariableOp*
_output_shapes

:*
dtype02
ReadVariableOp_1
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSliceReadVariableOp_1:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_4?
MatMul_1MatMulstrided_slice_2:output:0strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2

MatMul_1?
ReadVariableOp_2ReadVariableOpreadvariableop_resource^ReadVariableOp_1*
_output_shapes

:*
dtype02
ReadVariableOp_2
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack?
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_5/stack_1?
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2?
strided_slice_5StridedSliceReadVariableOp_2:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_5?
MatMul_2MatMulstrided_slice_2:output:0strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2

MatMul_2z
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype02
ReadVariableOp_3x
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_6/stack|
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack_1|
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_3:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slice_6{
BiasAddBiasAddMatMul:product:0strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource^ReadVariableOp_3*
_output_shapes
:*
dtype02
ReadVariableOp_4x
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_7/stack|
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_7/stack_1|
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_7/stack_2?
strided_slice_7StridedSliceReadVariableOp_4:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_7?
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_1?
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource^ReadVariableOp_4*
_output_shapes
:*
dtype02
ReadVariableOp_5x
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack|
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_8/stack_1|
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack_2?
strided_slice_8StridedSliceReadVariableOp_5:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
strided_slice_8?
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_2~
ReadVariableOp_6ReadVariableOpreadvariableop_6_resource*
_output_shapes

:*
dtype02
ReadVariableOp_6
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_9/stack?
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stack_1?
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack_2?
strided_slice_9StridedSliceReadVariableOp_6:value:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_9z
MatMul_3MatMulzeros:output:0strided_slice_9:output:0*
T0*'
_output_shapes
:?????????2

MatMul_3?
ReadVariableOp_7ReadVariableOpreadvariableop_6_resource^ReadVariableOp_6*
_output_shapes

:*
dtype02
ReadVariableOp_7?
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_10/stack?
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_10/stack_1?
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_10/stack_2?
strided_slice_10StridedSliceReadVariableOp_7:value:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_10{
MatMul_4MatMulzeros:output:0strided_slice_10:output:0*
T0*'
_output_shapes
:?????????2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1\
MulMuladd:z:0Const:output:0*
T0*'
_output_shapes
:?????????2
Mulb
Add_1AddMul:z:0Const_1:output:0*
T0*'
_output_shapes
:?????????2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_valueq
add_2AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3d
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*'
_output_shapes
:?????????2
Mul_1d
Add_3Add	Mul_1:z:0Const_3:output:0*
T0*'
_output_shapes
:?????????2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1l
mul_2Mulclip_by_value_1:z:0zeros:output:0*
T0*'
_output_shapes
:?????????2
mul_2?
ReadVariableOp_8ReadVariableOpreadvariableop_6_resource^ReadVariableOp_7*
_output_shapes

:*
dtype02
ReadVariableOp_8?
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_11/stack?
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_11/stack_1?
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_11/stack_2?
strided_slice_11StridedSliceReadVariableOp_8:value:0strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_11v
MatMul_5MatMul	mul_2:z:0strided_slice_11:output:0*
T0*'
_output_shapes
:?????????2

MatMul_5q
add_4AddV2BiasAdd_2:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
add_4Q
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:?????????2
Tanhj
mul_3Mulclip_by_value:z:0zeros:output:0*
T0*'
_output_shapes
:?????????2
mul_3S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xf
subSubsub/x:output:0clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
subZ
mul_4Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_4_
add_5AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:?????????2
add_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0readvariableop_resourcereadvariableop_3_resourcereadvariableop_6_resource^ReadVariableOp_2^ReadVariableOp_5^ReadVariableOp_8*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????: : : : : *
bodyR
while_body_680271*
condR
while_cond_680270*8
output_shapes'
%: : : : :?????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_12/stack~
strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_12/stack_1~
strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_12/stack_2?
strided_slice_12StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_12y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_1?
IdentityIdentitystrided_slice_12:output:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8^while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_82
whilewhile:( $
"
_user_specified_name
inputs/0
?
_
C__inference_reshape_layer_call_and_return_conditional_losses_679205

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapes
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????2	
Reshapeh
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?;
?
?__inference_gru_layer_call_and_return_conditional_losses_678468

inputs"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity??StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*:
_output_shapes(
&:?????????:?????????**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_gru_cell_layer_call_and_return_conditional_losses_6780832
StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4^StatefulPartitionedCall*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????: : : : : *
bodyR
while_body_678408*
condR
while_cond_678407*8
output_shapes'
%: : : : :?????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_1?
IdentityIdentitystrided_slice_3:output:0^StatefulPartitionedCall^while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall2
whilewhile:& "
 
_user_specified_nameinputs
?
?
A__inference_model_layer_call_and_return_conditional_losses_679285

inputs
inputs_1&
"gru_statefulpartitionedcall_args_1&
"gru_statefulpartitionedcall_args_2&
"gru_statefulpartitionedcall_args_3(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?gru/StatefulPartitionedCall?
gru/StatefulPartitionedCallStatefulPartitionedCallinputs_1"gru_statefulpartitionedcall_args_1"gru_statefulpartitionedcall_args_2"gru_statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_6790982
gru/StatefulPartitionedCall?
concatenate/PartitionedCallPartitionedCall$gru/StatefulPartitionedCall:output:0inputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????(**
config_proto

CPU

GPU 2J 8*P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_6791262
concatenate/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????(**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_6791452
dense/StatefulPartitionedCall?
re_lu/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????(**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_6791622
re_lu/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_6791802!
dense_1/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_6792052
reshape/PartitionedCall?
IdentityIdentity reshape/PartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^gru/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:????????? :?????????:::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
?
?
$__inference_gru_layer_call_fn_680945

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_6788422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
]
A__inference_re_lu_layer_call_and_return_conditional_losses_679162

inputs
identityN
ReluReluinputs*
T0*'
_output_shapes
:?????????(2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????(:& "
 
_user_specified_nameinputs
??
?
!__inference__wrapped_model_677962
input_2
input_1%
!model_gru_readvariableop_resource'
#model_gru_readvariableop_3_resource'
#model_gru_readvariableop_6_resource.
*model_dense_matmul_readvariableop_resource/
+model_dense_biasadd_readvariableop_resource0
,model_dense_1_matmul_readvariableop_resource1
-model_dense_1_biasadd_readvariableop_resource
identity??"model/dense/BiasAdd/ReadVariableOp?!model/dense/MatMul/ReadVariableOp?$model/dense_1/BiasAdd/ReadVariableOp?#model/dense_1/MatMul/ReadVariableOp?model/gru/ReadVariableOp?model/gru/ReadVariableOp_1?model/gru/ReadVariableOp_2?model/gru/ReadVariableOp_3?model/gru/ReadVariableOp_4?model/gru/ReadVariableOp_5?model/gru/ReadVariableOp_6?model/gru/ReadVariableOp_7?model/gru/ReadVariableOp_8?model/gru/whileY
model/gru/ShapeShapeinput_1*
T0*
_output_shapes
:2
model/gru/Shape?
model/gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
model/gru/strided_slice/stack?
model/gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
model/gru/strided_slice/stack_1?
model/gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
model/gru/strided_slice/stack_2?
model/gru/strided_sliceStridedSlicemodel/gru/Shape:output:0&model/gru/strided_slice/stack:output:0(model/gru/strided_slice/stack_1:output:0(model/gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model/gru/strided_slicep
model/gru/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
model/gru/zeros/mul/y?
model/gru/zeros/mulMul model/gru/strided_slice:output:0model/gru/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
model/gru/zeros/muls
model/gru/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
model/gru/zeros/Less/y?
model/gru/zeros/LessLessmodel/gru/zeros/mul:z:0model/gru/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
model/gru/zeros/Lessv
model/gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
model/gru/zeros/packed/1?
model/gru/zeros/packedPack model/gru/strided_slice:output:0!model/gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
model/gru/zeros/packeds
model/gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model/gru/zeros/Const?
model/gru/zerosFillmodel/gru/zeros/packed:output:0model/gru/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
model/gru/zeros?
model/gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
model/gru/transpose/perm?
model/gru/transpose	Transposeinput_1!model/gru/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
model/gru/transposem
model/gru/Shape_1Shapemodel/gru/transpose:y:0*
T0*
_output_shapes
:2
model/gru/Shape_1?
model/gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
model/gru/strided_slice_1/stack?
!model/gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!model/gru/strided_slice_1/stack_1?
!model/gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!model/gru/strided_slice_1/stack_2?
model/gru/strided_slice_1StridedSlicemodel/gru/Shape_1:output:0(model/gru/strided_slice_1/stack:output:0*model/gru/strided_slice_1/stack_1:output:0*model/gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model/gru/strided_slice_1?
%model/gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%model/gru/TensorArrayV2/element_shape?
model/gru/TensorArrayV2TensorListReserve.model/gru/TensorArrayV2/element_shape:output:0"model/gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
model/gru/TensorArrayV2?
?model/gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2A
?model/gru/TensorArrayUnstack/TensorListFromTensor/element_shape?
1model/gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel/gru/transpose:y:0Hmodel/gru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type023
1model/gru/TensorArrayUnstack/TensorListFromTensor?
model/gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
model/gru/strided_slice_2/stack?
!model/gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!model/gru/strided_slice_2/stack_1?
!model/gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!model/gru/strided_slice_2/stack_2?
model/gru/strided_slice_2StridedSlicemodel/gru/transpose:y:0(model/gru/strided_slice_2/stack:output:0*model/gru/strided_slice_2/stack_1:output:0*model/gru/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
model/gru/strided_slice_2?
model/gru/ReadVariableOpReadVariableOp!model_gru_readvariableop_resource*
_output_shapes

:*
dtype02
model/gru/ReadVariableOp?
model/gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
model/gru/strided_slice_3/stack?
!model/gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!model/gru/strided_slice_3/stack_1?
!model/gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!model/gru/strided_slice_3/stack_2?
model/gru/strided_slice_3StridedSlice model/gru/ReadVariableOp:value:0(model/gru/strided_slice_3/stack:output:0*model/gru/strided_slice_3/stack_1:output:0*model/gru/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
model/gru/strided_slice_3?
model/gru/MatMulMatMul"model/gru/strided_slice_2:output:0"model/gru/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
model/gru/MatMul?
model/gru/ReadVariableOp_1ReadVariableOp!model_gru_readvariableop_resource^model/gru/ReadVariableOp*
_output_shapes

:*
dtype02
model/gru/ReadVariableOp_1?
model/gru/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
model/gru/strided_slice_4/stack?
!model/gru/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!model/gru/strided_slice_4/stack_1?
!model/gru/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!model/gru/strided_slice_4/stack_2?
model/gru/strided_slice_4StridedSlice"model/gru/ReadVariableOp_1:value:0(model/gru/strided_slice_4/stack:output:0*model/gru/strided_slice_4/stack_1:output:0*model/gru/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
model/gru/strided_slice_4?
model/gru/MatMul_1MatMul"model/gru/strided_slice_2:output:0"model/gru/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2
model/gru/MatMul_1?
model/gru/ReadVariableOp_2ReadVariableOp!model_gru_readvariableop_resource^model/gru/ReadVariableOp_1*
_output_shapes

:*
dtype02
model/gru/ReadVariableOp_2?
model/gru/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
model/gru/strided_slice_5/stack?
!model/gru/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!model/gru/strided_slice_5/stack_1?
!model/gru/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!model/gru/strided_slice_5/stack_2?
model/gru/strided_slice_5StridedSlice"model/gru/ReadVariableOp_2:value:0(model/gru/strided_slice_5/stack:output:0*model/gru/strided_slice_5/stack_1:output:0*model/gru/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
model/gru/strided_slice_5?
model/gru/MatMul_2MatMul"model/gru/strided_slice_2:output:0"model/gru/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
model/gru/MatMul_2?
model/gru/ReadVariableOp_3ReadVariableOp#model_gru_readvariableop_3_resource*
_output_shapes
:*
dtype02
model/gru/ReadVariableOp_3?
model/gru/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
model/gru/strided_slice_6/stack?
!model/gru/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!model/gru/strided_slice_6/stack_1?
!model/gru/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!model/gru/strided_slice_6/stack_2?
model/gru/strided_slice_6StridedSlice"model/gru/ReadVariableOp_3:value:0(model/gru/strided_slice_6/stack:output:0*model/gru/strided_slice_6/stack_1:output:0*model/gru/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
model/gru/strided_slice_6?
model/gru/BiasAddBiasAddmodel/gru/MatMul:product:0"model/gru/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2
model/gru/BiasAdd?
model/gru/ReadVariableOp_4ReadVariableOp#model_gru_readvariableop_3_resource^model/gru/ReadVariableOp_3*
_output_shapes
:*
dtype02
model/gru/ReadVariableOp_4?
model/gru/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:2!
model/gru/strided_slice_7/stack?
!model/gru/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!model/gru/strided_slice_7/stack_1?
!model/gru/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!model/gru/strided_slice_7/stack_2?
model/gru/strided_slice_7StridedSlice"model/gru/ReadVariableOp_4:value:0(model/gru/strided_slice_7/stack:output:0*model/gru/strided_slice_7/stack_1:output:0*model/gru/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
model/gru/strided_slice_7?
model/gru/BiasAdd_1BiasAddmodel/gru/MatMul_1:product:0"model/gru/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2
model/gru/BiasAdd_1?
model/gru/ReadVariableOp_5ReadVariableOp#model_gru_readvariableop_3_resource^model/gru/ReadVariableOp_4*
_output_shapes
:*
dtype02
model/gru/ReadVariableOp_5?
model/gru/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB:2!
model/gru/strided_slice_8/stack?
!model/gru/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!model/gru/strided_slice_8/stack_1?
!model/gru/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!model/gru/strided_slice_8/stack_2?
model/gru/strided_slice_8StridedSlice"model/gru/ReadVariableOp_5:value:0(model/gru/strided_slice_8/stack:output:0*model/gru/strided_slice_8/stack_1:output:0*model/gru/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
model/gru/strided_slice_8?
model/gru/BiasAdd_2BiasAddmodel/gru/MatMul_2:product:0"model/gru/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2
model/gru/BiasAdd_2?
model/gru/ReadVariableOp_6ReadVariableOp#model_gru_readvariableop_6_resource*
_output_shapes

:*
dtype02
model/gru/ReadVariableOp_6?
model/gru/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
model/gru/strided_slice_9/stack?
!model/gru/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!model/gru/strided_slice_9/stack_1?
!model/gru/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!model/gru/strided_slice_9/stack_2?
model/gru/strided_slice_9StridedSlice"model/gru/ReadVariableOp_6:value:0(model/gru/strided_slice_9/stack:output:0*model/gru/strided_slice_9/stack_1:output:0*model/gru/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
model/gru/strided_slice_9?
model/gru/MatMul_3MatMulmodel/gru/zeros:output:0"model/gru/strided_slice_9:output:0*
T0*'
_output_shapes
:?????????2
model/gru/MatMul_3?
model/gru/ReadVariableOp_7ReadVariableOp#model_gru_readvariableop_6_resource^model/gru/ReadVariableOp_6*
_output_shapes

:*
dtype02
model/gru/ReadVariableOp_7?
 model/gru/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"       2"
 model/gru/strided_slice_10/stack?
"model/gru/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"model/gru/strided_slice_10/stack_1?
"model/gru/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"model/gru/strided_slice_10/stack_2?
model/gru/strided_slice_10StridedSlice"model/gru/ReadVariableOp_7:value:0)model/gru/strided_slice_10/stack:output:0+model/gru/strided_slice_10/stack_1:output:0+model/gru/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
model/gru/strided_slice_10?
model/gru/MatMul_4MatMulmodel/gru/zeros:output:0#model/gru/strided_slice_10:output:0*
T0*'
_output_shapes
:?????????2
model/gru/MatMul_4?
model/gru/addAddV2model/gru/BiasAdd:output:0model/gru/MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
model/gru/addg
model/gru/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
model/gru/Constk
model/gru/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
model/gru/Const_1?
model/gru/MulMulmodel/gru/add:z:0model/gru/Const:output:0*
T0*'
_output_shapes
:?????????2
model/gru/Mul?
model/gru/Add_1Addmodel/gru/Mul:z:0model/gru/Const_1:output:0*
T0*'
_output_shapes
:?????????2
model/gru/Add_1?
!model/gru/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!model/gru/clip_by_value/Minimum/y?
model/gru/clip_by_value/MinimumMinimummodel/gru/Add_1:z:0*model/gru/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2!
model/gru/clip_by_value/Minimum{
model/gru/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model/gru/clip_by_value/y?
model/gru/clip_by_valueMaximum#model/gru/clip_by_value/Minimum:z:0"model/gru/clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
model/gru/clip_by_value?
model/gru/add_2AddV2model/gru/BiasAdd_1:output:0model/gru/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
model/gru/add_2k
model/gru/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2
model/gru/Const_2k
model/gru/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
model/gru/Const_3?
model/gru/Mul_1Mulmodel/gru/add_2:z:0model/gru/Const_2:output:0*
T0*'
_output_shapes
:?????????2
model/gru/Mul_1?
model/gru/Add_3Addmodel/gru/Mul_1:z:0model/gru/Const_3:output:0*
T0*'
_output_shapes
:?????????2
model/gru/Add_3?
#model/gru/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#model/gru/clip_by_value_1/Minimum/y?
!model/gru/clip_by_value_1/MinimumMinimummodel/gru/Add_3:z:0,model/gru/clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2#
!model/gru/clip_by_value_1/Minimum
model/gru/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model/gru/clip_by_value_1/y?
model/gru/clip_by_value_1Maximum%model/gru/clip_by_value_1/Minimum:z:0$model/gru/clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2
model/gru/clip_by_value_1?
model/gru/mul_2Mulmodel/gru/clip_by_value_1:z:0model/gru/zeros:output:0*
T0*'
_output_shapes
:?????????2
model/gru/mul_2?
model/gru/ReadVariableOp_8ReadVariableOp#model_gru_readvariableop_6_resource^model/gru/ReadVariableOp_7*
_output_shapes

:*
dtype02
model/gru/ReadVariableOp_8?
 model/gru/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB"       2"
 model/gru/strided_slice_11/stack?
"model/gru/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2$
"model/gru/strided_slice_11/stack_1?
"model/gru/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"model/gru/strided_slice_11/stack_2?
model/gru/strided_slice_11StridedSlice"model/gru/ReadVariableOp_8:value:0)model/gru/strided_slice_11/stack:output:0+model/gru/strided_slice_11/stack_1:output:0+model/gru/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
model/gru/strided_slice_11?
model/gru/MatMul_5MatMulmodel/gru/mul_2:z:0#model/gru/strided_slice_11:output:0*
T0*'
_output_shapes
:?????????2
model/gru/MatMul_5?
model/gru/add_4AddV2model/gru/BiasAdd_2:output:0model/gru/MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
model/gru/add_4o
model/gru/TanhTanhmodel/gru/add_4:z:0*
T0*'
_output_shapes
:?????????2
model/gru/Tanh?
model/gru/mul_3Mulmodel/gru/clip_by_value:z:0model/gru/zeros:output:0*
T0*'
_output_shapes
:?????????2
model/gru/mul_3g
model/gru/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
model/gru/sub/x?
model/gru/subSubmodel/gru/sub/x:output:0model/gru/clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
model/gru/sub?
model/gru/mul_4Mulmodel/gru/sub:z:0model/gru/Tanh:y:0*
T0*'
_output_shapes
:?????????2
model/gru/mul_4?
model/gru/add_5AddV2model/gru/mul_3:z:0model/gru/mul_4:z:0*
T0*'
_output_shapes
:?????????2
model/gru/add_5?
'model/gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2)
'model/gru/TensorArrayV2_1/element_shape?
model/gru/TensorArrayV2_1TensorListReserve0model/gru/TensorArrayV2_1/element_shape:output:0"model/gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
model/gru/TensorArrayV2_1b
model/gru/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
model/gru/time?
"model/gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"model/gru/while/maximum_iterations~
model/gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
model/gru/while/loop_counter?
model/gru/whileWhile%model/gru/while/loop_counter:output:0+model/gru/while/maximum_iterations:output:0model/gru/time:output:0"model/gru/TensorArrayV2_1:handle:0model/gru/zeros:output:0"model/gru/strided_slice_1:output:0Amodel/gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0!model_gru_readvariableop_resource#model_gru_readvariableop_3_resource#model_gru_readvariableop_6_resource^model/gru/ReadVariableOp_2^model/gru/ReadVariableOp_5^model/gru/ReadVariableOp_8*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????: : : : : *'
bodyR
model_gru_while_body_677800*'
condR
model_gru_while_cond_677799*8
output_shapes'
%: : : : :?????????: : : : : *
parallel_iterations 2
model/gru/while?
:model/gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2<
:model/gru/TensorArrayV2Stack/TensorListStack/element_shape?
,model/gru/TensorArrayV2Stack/TensorListStackTensorListStackmodel/gru/while:output:3Cmodel/gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype02.
,model/gru/TensorArrayV2Stack/TensorListStack?
 model/gru/strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2"
 model/gru/strided_slice_12/stack?
"model/gru/strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"model/gru/strided_slice_12/stack_1?
"model/gru/strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"model/gru/strided_slice_12/stack_2?
model/gru/strided_slice_12StridedSlice5model/gru/TensorArrayV2Stack/TensorListStack:tensor:0)model/gru/strided_slice_12/stack:output:0+model/gru/strided_slice_12/stack_1:output:0+model/gru/strided_slice_12/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
model/gru/strided_slice_12?
model/gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
model/gru/transpose_1/perm?
model/gru/transpose_1	Transpose5model/gru/TensorArrayV2Stack/TensorListStack:tensor:0#model/gru/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????2
model/gru/transpose_1?
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/concatenate/concat/axis?
model/concatenate/concatConcatV2#model/gru/strided_slice_12:output:0input_2&model/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????(2
model/concatenate/concat?
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes

:((*
dtype02#
!model/dense/MatMul/ReadVariableOp?
model/dense/MatMulMatMul!model/concatenate/concat:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model/dense/MatMul?
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02$
"model/dense/BiasAdd/ReadVariableOp?
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
model/dense/BiasAdd|
model/re_lu/ReluRelumodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????(2
model/re_lu/Relu?
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes

:(*
dtype02%
#model/dense_1/MatMul/ReadVariableOp?
model/dense_1/MatMulMatMulmodel/re_lu/Relu:activations:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/dense_1/MatMul?
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/dense_1/BiasAdd/ReadVariableOp?
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/dense_1/BiasAddx
model/reshape/ShapeShapemodel/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:2
model/reshape/Shape?
!model/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!model/reshape/strided_slice/stack?
#model/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#model/reshape/strided_slice/stack_1?
#model/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#model/reshape/strided_slice/stack_2?
model/reshape/strided_sliceStridedSlicemodel/reshape/Shape:output:0*model/reshape/strided_slice/stack:output:0,model/reshape/strided_slice/stack_1:output:0,model/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model/reshape/strided_slice?
model/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
model/reshape/Reshape/shape/1?
model/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
model/reshape/Reshape/shape/2?
model/reshape/Reshape/shapePack$model/reshape/strided_slice:output:0&model/reshape/Reshape/shape/1:output:0&model/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
model/reshape/Reshape/shape?
model/reshape/ReshapeReshapemodel/dense_1/BiasAdd:output:0$model/reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????2
model/reshape/Reshape?
IdentityIdentitymodel/reshape/Reshape:output:0#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp^model/gru/ReadVariableOp^model/gru/ReadVariableOp_1^model/gru/ReadVariableOp_2^model/gru/ReadVariableOp_3^model/gru/ReadVariableOp_4^model/gru/ReadVariableOp_5^model/gru/ReadVariableOp_6^model/gru/ReadVariableOp_7^model/gru/ReadVariableOp_8^model/gru/while*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:????????? :?????????:::::::2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp24
model/gru/ReadVariableOpmodel/gru/ReadVariableOp28
model/gru/ReadVariableOp_1model/gru/ReadVariableOp_128
model/gru/ReadVariableOp_2model/gru/ReadVariableOp_228
model/gru/ReadVariableOp_3model/gru/ReadVariableOp_328
model/gru/ReadVariableOp_4model/gru/ReadVariableOp_428
model/gru/ReadVariableOp_5model/gru/ReadVariableOp_528
model/gru/ReadVariableOp_6model/gru/ReadVariableOp_628
model/gru/ReadVariableOp_7model/gru/ReadVariableOp_728
model/gru/ReadVariableOp_8model/gru/ReadVariableOp_82"
model/gru/whilemodel/gru/while:' #
!
_user_specified_name	input_2:'#
!
_user_specified_name	input_1
?
?
while_cond_678515
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
less_strided_slice_1.
*while_cond_678515___redundant_placeholder0.
*while_cond_678515___redundant_placeholder1.
*while_cond_678515___redundant_placeholder2.
*while_cond_678515___redundant_placeholder3
identity
X
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: 2
LessK
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-: : : : :?????????: ::::
?
?
$__inference_gru_layer_call_fn_680417
inputs_0"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_6784682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0
?
s
G__inference_concatenate_layer_call_and_return_conditional_losses_680960
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????(2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:????????? :( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
?
]
A__inference_re_lu_layer_call_and_return_conditional_losses_680988

inputs
identityN
ReluReluinputs*
T0*'
_output_shapes
:?????????(2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????(:& "
 
_user_specified_nameinputs
?
?
C__inference_dense_1_layer_call_and_return_conditional_losses_679180

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????(::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
?
D
(__inference_reshape_layer_call_fn_681028

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_6792052
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:& "
 
_user_specified_nameinputs
?
?
$__inference_gru_layer_call_fn_680953

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*H
fCRA
?__inference_gru_layer_call_and_return_conditional_losses_6790982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?

?
&__inference_model_layer_call_fn_679295
input_2
input_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2input_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*+
_output_shapes
:?????????**
config_proto

CPU

GPU 2J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_6792852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:????????? :?????????:::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_2:'#
!
_user_specified_name	input_1
?$
?
"__inference__traced_restore_681307
file_prefix#
assignvariableop_dense_2_kernel#
assignvariableop_1_dense_2_bias'
#assignvariableop_2_dense_1_1_kernel%
!assignvariableop_3_dense_1_1_bias#
assignvariableop_4_gru_1_kernel-
)assignvariableop_5_gru_1_recurrent_kernel!
assignvariableop_6_gru_1_bias

identity_8??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	22
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_dense_2_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_2_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_1_1_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_1_1_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_gru_1_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp)assignvariableop_5_gru_1_recurrent_kernelIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_gru_1_biasIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names?
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_7Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_7?

Identity_8IdentityIdentity_7:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2

Identity_8"!

identity_8Identity_8:output:0*1
_input_shapes 
: :::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
?	
?
)__inference_gru_cell_layer_call_fn_681228

inputs
states_0"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*:
_output_shapes(
&:?????????:?????????**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_gru_cell_layer_call_and_return_conditional_losses_6781722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*E
_input_shapes4
2:?????????:?????????:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0
?
?
A__inference_dense_layer_call_and_return_conditional_losses_679145

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:((*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????(2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????(::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
ے
?
?__inference_gru_layer_call_and_return_conditional_losses_679098

inputs
readvariableop_resource
readvariableop_3_resource
readvariableop_6_resource
identity??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?ReadVariableOp_4?ReadVariableOp_5?ReadVariableOp_6?ReadVariableOp_7?ReadVariableOp_8?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2x
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype02
ReadVariableOp
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_3?
MatMulMatMulstrided_slice_2:output:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2
MatMul?
ReadVariableOp_1ReadVariableOpreadvariableop_resource^ReadVariableOp*
_output_shapes

:*
dtype02
ReadVariableOp_1
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSliceReadVariableOp_1:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_4?
MatMul_1MatMulstrided_slice_2:output:0strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2

MatMul_1?
ReadVariableOp_2ReadVariableOpreadvariableop_resource^ReadVariableOp_1*
_output_shapes

:*
dtype02
ReadVariableOp_2
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack?
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_5/stack_1?
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2?
strided_slice_5StridedSliceReadVariableOp_2:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_5?
MatMul_2MatMulstrided_slice_2:output:0strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2

MatMul_2z
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource*
_output_shapes
:*
dtype02
ReadVariableOp_3x
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_6/stack|
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack_1|
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_3:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slice_6{
BiasAddBiasAddMatMul:product:0strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource^ReadVariableOp_3*
_output_shapes
:*
dtype02
ReadVariableOp_4x
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_7/stack|
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_7/stack_1|
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_7/stack_2?
strided_slice_7StridedSliceReadVariableOp_4:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_7?
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_1?
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource^ReadVariableOp_4*
_output_shapes
:*
dtype02
ReadVariableOp_5x
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack|
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_8/stack_1|
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack_2?
strided_slice_8StridedSliceReadVariableOp_5:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
strided_slice_8?
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_2~
ReadVariableOp_6ReadVariableOpreadvariableop_6_resource*
_output_shapes

:*
dtype02
ReadVariableOp_6
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_9/stack?
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stack_1?
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack_2?
strided_slice_9StridedSliceReadVariableOp_6:value:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_9z
MatMul_3MatMulzeros:output:0strided_slice_9:output:0*
T0*'
_output_shapes
:?????????2

MatMul_3?
ReadVariableOp_7ReadVariableOpreadvariableop_6_resource^ReadVariableOp_6*
_output_shapes

:*
dtype02
ReadVariableOp_7?
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_10/stack?
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_10/stack_1?
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_10/stack_2?
strided_slice_10StridedSliceReadVariableOp_7:value:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_10{
MatMul_4MatMulzeros:output:0strided_slice_10:output:0*
T0*'
_output_shapes
:?????????2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1\
MulMuladd:z:0Const:output:0*
T0*'
_output_shapes
:?????????2
Mulb
Add_1AddMul:z:0Const_1:output:0*
T0*'
_output_shapes
:?????????2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_valueq
add_2AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3d
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*'
_output_shapes
:?????????2
Mul_1d
Add_3Add	Mul_1:z:0Const_3:output:0*
T0*'
_output_shapes
:?????????2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1l
mul_2Mulclip_by_value_1:z:0zeros:output:0*
T0*'
_output_shapes
:?????????2
mul_2?
ReadVariableOp_8ReadVariableOpreadvariableop_6_resource^ReadVariableOp_7*
_output_shapes

:*
dtype02
ReadVariableOp_8?
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_11/stack?
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_11/stack_1?
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_11/stack_2?
strided_slice_11StridedSliceReadVariableOp_8:value:0strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_11v
MatMul_5MatMul	mul_2:z:0strided_slice_11:output:0*
T0*'
_output_shapes
:?????????2

MatMul_5q
add_4AddV2BiasAdd_2:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
add_4Q
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:?????????2
Tanhj
mul_3Mulclip_by_value:z:0zeros:output:0*
T0*'
_output_shapes
:?????????2
mul_3S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xf
subSubsub/x:output:0clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
subZ
mul_4Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_4_
add_5AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:?????????2
add_5?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0readvariableop_resourcereadvariableop_3_resourcereadvariableop_6_resource^ReadVariableOp_2^ReadVariableOp_5^ReadVariableOp_8*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????: : : : : *
bodyR
while_body_678960*
condR
while_cond_678959*8
output_shapes'
%: : : : :?????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_12/stack~
strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_12/stack_1~
strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_12/stack_2?
strided_slice_12StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_12y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????2
transpose_1?
IdentityIdentitystrided_slice_12:output:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8^while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_82
whilewhile:& "
 
_user_specified_nameinputs
?v
?
while_body_680543
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
readvariableop_resource_0
readvariableop_3_resource_0
readvariableop_6_resource_0
identity

identity_1

identity_2

identity_3

identity_4
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
readvariableop_resource
readvariableop_3_resource
readvariableop_6_resource??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?ReadVariableOp_4?ReadVariableOp_5?ReadVariableOp_6?ReadVariableOp_7?ReadVariableOp_8?
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   23
1TensorArrayV2Read/TensorListGetItem/element_shape?
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02%
#TensorArrayV2Read/TensorListGetItemz
ReadVariableOpReadVariableOpreadvariableop_resource_0*
_output_shapes

:*
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice?
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice:output:0*
T0*'
_output_shapes
:?????????2
MatMul?
ReadVariableOp_1ReadVariableOpreadvariableop_resource_0^ReadVariableOp*
_output_shapes

:*
dtype02
ReadVariableOp_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_1:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_2?
MatMul_1MatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2

MatMul_1?
ReadVariableOp_2ReadVariableOpreadvariableop_resource_0^ReadVariableOp_1*
_output_shapes

:*
dtype02
ReadVariableOp_2
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_2:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_3?
MatMul_2MatMul*TensorArrayV2Read/TensorListGetItem:item:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????2

MatMul_2|
ReadVariableOp_3ReadVariableOpreadvariableop_3_resource_0*
_output_shapes
:*
dtype02
ReadVariableOp_3x
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_4/stack|
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2?
strided_slice_4StridedSliceReadVariableOp_3:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slice_4{
BiasAddBiasAddMatMul:product:0strided_slice_4:output:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
ReadVariableOp_4ReadVariableOpreadvariableop_3_resource_0^ReadVariableOp_3*
_output_shapes
:*
dtype02
ReadVariableOp_4x
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack|
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_1|
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_2?
strided_slice_5StridedSliceReadVariableOp_4:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_5?
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_5:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_1?
ReadVariableOp_5ReadVariableOpreadvariableop_3_resource_0^ReadVariableOp_4*
_output_shapes
:*
dtype02
ReadVariableOp_5x
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack|
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_6/stack_1|
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_5:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
strided_slice_6?
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_6:output:0*
T0*'
_output_shapes
:?????????2
	BiasAdd_2?
ReadVariableOp_6ReadVariableOpreadvariableop_6_resource_0*
_output_shapes

:*
dtype02
ReadVariableOp_6
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSliceReadVariableOp_6:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_7y
MatMul_3MatMulplaceholder_2strided_slice_7:output:0*
T0*'
_output_shapes
:?????????2

MatMul_3?
ReadVariableOp_7ReadVariableOpreadvariableop_6_resource_0^ReadVariableOp_6*
_output_shapes

:*
dtype02
ReadVariableOp_7
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack?
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_8/stack_1?
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2?
strided_slice_8StridedSliceReadVariableOp_7:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_8y
MatMul_4MatMulplaceholder_2strided_slice_8:output:0*
T0*'
_output_shapes
:?????????2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_3:product:0*
T0*'
_output_shapes
:?????????2
addS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
ConstW
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_1\
MulMuladd:z:0Const:output:0*
T0*'
_output_shapes
:?????????2
Mulb
Add_1AddMul:z:0Const_1:output:0*
T0*'
_output_shapes
:?????????2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_valueq
add_2AddV2BiasAdd_1:output:0MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
add_2W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *??L>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3d
Mul_1Mul	add_2:z:0Const_2:output:0*
T0*'
_output_shapes
:?????????2
Mul_1d
Add_3Add	Mul_1:z:0Const_3:output:0*
T0*'
_output_shapes
:?????????2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
clip_by_value_1/Minimum/y?
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y?
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*'
_output_shapes
:?????????2
clip_by_value_1k
mul_2Mulclip_by_value_1:z:0placeholder_2*
T0*'
_output_shapes
:?????????2
mul_2?
ReadVariableOp_8ReadVariableOpreadvariableop_6_resource_0^ReadVariableOp_7*
_output_shapes

:*
dtype02
ReadVariableOp_8
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_9/stack?
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_9/stack_1?
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack_2?
strided_slice_9StridedSliceReadVariableOp_8:value:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask2
strided_slice_9u
MatMul_5MatMul	mul_2:z:0strided_slice_9:output:0*
T0*'
_output_shapes
:?????????2

MatMul_5q
add_4AddV2BiasAdd_2:output:0MatMul_5:product:0*
T0*'
_output_shapes
:?????????2
add_4Q
TanhTanh	add_4:z:0*
T0*'
_output_shapes
:?????????2
Tanhi
mul_3Mulclip_by_value:z:0placeholder_2*
T0*'
_output_shapes
:?????????2
mul_3S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xf
subSubsub/x:output:0clip_by_value:z:0*
T0*'
_output_shapes
:?????????2
subZ
mul_4Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????2
mul_4_
add_5AddV2	mul_3:z:0	mul_4:z:0*
T0*'
_output_shapes
:?????????2
add_5?
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	add_5:z:0*
_output_shapes
: *
element_dtype02&
$TensorArrayV2Write/TensorListSetItemT
add_6/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_6/yW
add_6AddV2placeholderadd_6/y:output:0*
T0*
_output_shapes
: 2
add_6T
add_7/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_7/y^
add_7AddV2while_loop_counteradd_7/y:output:0*
T0*
_output_shapes
: 2
add_7?
IdentityIdentity	add_7:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: 2

Identity?

Identity_1Identitywhile_maximum_iterations^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: 2

Identity_1?

Identity_2Identity	add_6:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*
_output_shapes
: 2

Identity_3?

Identity_4Identity	add_5:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8*
T0*'
_output_shapes
:?????????2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"8
readvariableop_3_resourcereadvariableop_3_resource_0"8
readvariableop_6_resourcereadvariableop_6_resource_0"4
readvariableop_resourcereadvariableop_resource_0"$
strided_slice_1strided_slice_1_0"?
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????: : :::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_8"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
?
input_14
serving_default_input_1:0?????????
;
input_20
serving_default_input_2:0????????? ?
reshape4
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?3
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	trainable_variables

regularization_losses
	variables
	keras_api

signatures
*S&call_and_return_all_conditional_losses
T_default_save_signature
U__call__"?0
_tf_keras_model?0{"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 8, 5], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "GRU", "config": {"name": "gru", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 8, "activation": "tanh", "recurrent_activation": "hard_sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 1, "reset_after": false}, "name": "gru", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": [null, 32], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": 1}, "name": "concatenate", "inbound_nodes": [[["gru", 0, 0, {}], ["input_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 40, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 5, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["re_lu", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": [1, 5]}, "name": "reshape", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0], ["input_1", 0, 0]], "output_layers": [["reshape", 0, 0]]}, "input_spec": [null, null], "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 8, 5], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "GRU", "config": {"name": "gru", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 8, "activation": "tanh", "recurrent_activation": "hard_sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 1, "reset_after": false}, "name": "gru", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": [null, 32], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": 1}, "name": "concatenate", "inbound_nodes": [[["gru", 0, 0, {}], ["input_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 40, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 5, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["re_lu", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": [1, 5]}, "name": "reshape", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0], ["input_1", 0, 0]], "output_layers": [["reshape", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 8, 5], "config": {"batch_input_shape": [null, 8, 5], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?

cell

state_spec
trainable_variables
regularization_losses
	variables
	keras_api
*V&call_and_return_all_conditional_losses
W__call__"?
_tf_keras_layer?{"class_name": "GRU", "name": "gru", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "gru", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 8, "activation": "tanh", "recurrent_activation": "hard_sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 1, "reset_after": false}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": [null, null, 5], "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}]}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 32], "config": {"batch_input_shape": [null, 32], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
?
trainable_variables
regularization_losses
	variables
	keras_api
*X&call_and_return_all_conditional_losses
Y__call__"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": 1}}
?

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*Z&call_and_return_all_conditional_losses
[__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 40, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40}}}}
?
trainable_variables
regularization_losses
 	variables
!	keras_api
*\&call_and_return_all_conditional_losses
]__call__"?
_tf_keras_layer?{"class_name": "ReLU", "name": "re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
?

"kernel
#bias
$trainable_variables
%regularization_losses
&	variables
'	keras_api
*^&call_and_return_all_conditional_losses
___call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 5, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 40}}}}
?
(trainable_variables
)regularization_losses
*	variables
+	keras_api
*`&call_and_return_all_conditional_losses
a__call__"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": [1, 5]}}
Q
,0
-1
.2
3
4
"5
#6"
trackable_list_wrapper
 "
trackable_list_wrapper
Q
,0
-1
.2
3
4
"5
#6"
trackable_list_wrapper
?
/metrics
0non_trainable_variables
	trainable_variables

regularization_losses

1layers
2layer_regularization_losses
	variables
U__call__
T_default_save_signature
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
,
bserving_default"
signature_map
?

,kernel
-recurrent_kernel
.bias
3trainable_variables
4regularization_losses
5	variables
6	keras_api
*c&call_and_return_all_conditional_losses
d__call__"?
_tf_keras_layer?{"class_name": "GRUCell", "name": "gru_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "gru_cell", "trainable": true, "dtype": "float32", "units": 8, "activation": "tanh", "recurrent_activation": "hard_sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 1, "reset_after": false}}
 "
trackable_list_wrapper
5
,0
-1
.2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
,0
-1
.2"
trackable_list_wrapper
?
7metrics
8non_trainable_variables
trainable_variables
regularization_losses

9layers
:layer_regularization_losses
	variables
W__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
;metrics
<non_trainable_variables
trainable_variables
regularization_losses

=layers
>layer_regularization_losses
	variables
Y__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
 :((2dense_2/kernel
:(2dense_2/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
?metrics
@non_trainable_variables
trainable_variables
regularization_losses

Alayers
Blayer_regularization_losses
	variables
[__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Cmetrics
Dnon_trainable_variables
trainable_variables
regularization_losses

Elayers
Flayer_regularization_losses
 	variables
]__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
": (2dense_1_1/kernel
:2dense_1_1/bias
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
?
Gmetrics
Hnon_trainable_variables
$trainable_variables
%regularization_losses

Ilayers
Jlayer_regularization_losses
&	variables
___call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Kmetrics
Lnon_trainable_variables
(trainable_variables
)regularization_losses

Mlayers
Nlayer_regularization_losses
*	variables
a__call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
:2gru_1/kernel
(:&2gru_1/recurrent_kernel
:2
gru_1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
5
,0
-1
.2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
,0
-1
.2"
trackable_list_wrapper
?
Ometrics
Pnon_trainable_variables
3trainable_variables
4regularization_losses

Qlayers
Rlayer_regularization_losses
5	variables
d__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
A__inference_model_layer_call_and_return_conditional_losses_679871
A__inference_model_layer_call_and_return_conditional_losses_679590
A__inference_model_layer_call_and_return_conditional_losses_679214
A__inference_model_layer_call_and_return_conditional_losses_679232?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
!__inference__wrapped_model_677962?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *R?O
M?J
!?
input_2????????? 
%?"
input_1?????????
?2?
&__inference_model_layer_call_fn_679264
&__inference_model_layer_call_fn_679897
&__inference_model_layer_call_fn_679884
&__inference_model_layer_call_fn_679295?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
?__inference_gru_layer_call_and_return_conditional_losses_680153
?__inference_gru_layer_call_and_return_conditional_losses_680681
?__inference_gru_layer_call_and_return_conditional_losses_680409
?__inference_gru_layer_call_and_return_conditional_losses_680937?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
$__inference_gru_layer_call_fn_680945
$__inference_gru_layer_call_fn_680425
$__inference_gru_layer_call_fn_680417
$__inference_gru_layer_call_fn_680953?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_concatenate_layer_call_and_return_conditional_losses_680960?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_concatenate_layer_call_fn_680966?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_layer_call_and_return_conditional_losses_680976?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense_layer_call_fn_680983?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_re_lu_layer_call_and_return_conditional_losses_680988?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_re_lu_layer_call_fn_680993?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_1_layer_call_and_return_conditional_losses_681003?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_1_layer_call_fn_681010?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_reshape_layer_call_and_return_conditional_losses_681023?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_reshape_layer_call_fn_681028?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
:B8
$__inference_signature_wrapper_679309input_1input_2
?2?
D__inference_gru_cell_layer_call_and_return_conditional_losses_681117
D__inference_gru_cell_layer_call_and_return_conditional_losses_681206?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_gru_cell_layer_call_fn_681228
)__inference_gru_cell_layer_call_fn_681217?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ?
!__inference__wrapped_model_677962?,.-"#\?Y
R?O
M?J
!?
input_2????????? 
%?"
input_1?????????
? "5?2
0
reshape%?"
reshape??????????
G__inference_concatenate_layer_call_and_return_conditional_losses_680960?Z?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1????????? 
? "%?"
?
0?????????(
? ?
,__inference_concatenate_layer_call_fn_680966vZ?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1????????? 
? "??????????(?
C__inference_dense_1_layer_call_and_return_conditional_losses_681003\"#/?,
%?"
 ?
inputs?????????(
? "%?"
?
0?????????
? {
(__inference_dense_1_layer_call_fn_681010O"#/?,
%?"
 ?
inputs?????????(
? "???????????
A__inference_dense_layer_call_and_return_conditional_losses_680976\/?,
%?"
 ?
inputs?????????(
? "%?"
?
0?????????(
? y
&__inference_dense_layer_call_fn_680983O/?,
%?"
 ?
inputs?????????(
? "??????????(?
D__inference_gru_cell_layer_call_and_return_conditional_losses_681117?,.-\?Y
R?O
 ?
inputs?????????
'?$
"?
states/0?????????
p
? "R?O
H?E
?
0/0?????????
$?!
?
0/1/0?????????
? ?
D__inference_gru_cell_layer_call_and_return_conditional_losses_681206?,.-\?Y
R?O
 ?
inputs?????????
'?$
"?
states/0?????????
p 
? "R?O
H?E
?
0/0?????????
$?!
?
0/1/0?????????
? ?
)__inference_gru_cell_layer_call_fn_681217?,.-\?Y
R?O
 ?
inputs?????????
'?$
"?
states/0?????????
p
? "D?A
?
0?????????
"?
?
1/0??????????
)__inference_gru_cell_layer_call_fn_681228?,.-\?Y
R?O
 ?
inputs?????????
'?$
"?
states/0?????????
p 
? "D?A
?
0?????????
"?
?
1/0??????????
?__inference_gru_layer_call_and_return_conditional_losses_680153},.-O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "%?"
?
0?????????
? ?
?__inference_gru_layer_call_and_return_conditional_losses_680409},.-O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "%?"
?
0?????????
? ?
?__inference_gru_layer_call_and_return_conditional_losses_680681m,.-??<
5?2
$?!
inputs?????????

 
p

 
? "%?"
?
0?????????
? ?
?__inference_gru_layer_call_and_return_conditional_losses_680937m,.-??<
5?2
$?!
inputs?????????

 
p 

 
? "%?"
?
0?????????
? ?
$__inference_gru_layer_call_fn_680417p,.-O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "???????????
$__inference_gru_layer_call_fn_680425p,.-O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "???????????
$__inference_gru_layer_call_fn_680945`,.-??<
5?2
$?!
inputs?????????

 
p

 
? "???????????
$__inference_gru_layer_call_fn_680953`,.-??<
5?2
$?!
inputs?????????

 
p 

 
? "???????????
A__inference_model_layer_call_and_return_conditional_losses_679214?,.-"#d?a
Z?W
M?J
!?
input_2????????? 
%?"
input_1?????????
p

 
? ")?&
?
0?????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_679232?,.-"#d?a
Z?W
M?J
!?
input_2????????? 
%?"
input_1?????????
p 

 
? ")?&
?
0?????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_679590?,.-"#f?c
\?Y
O?L
"?
inputs/0????????? 
&?#
inputs/1?????????
p

 
? ")?&
?
0?????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_679871?,.-"#f?c
\?Y
O?L
"?
inputs/0????????? 
&?#
inputs/1?????????
p 

 
? ")?&
?
0?????????
? ?
&__inference_model_layer_call_fn_679264?,.-"#d?a
Z?W
M?J
!?
input_2????????? 
%?"
input_1?????????
p

 
? "???????????
&__inference_model_layer_call_fn_679295?,.-"#d?a
Z?W
M?J
!?
input_2????????? 
%?"
input_1?????????
p 

 
? "???????????
&__inference_model_layer_call_fn_679884?,.-"#f?c
\?Y
O?L
"?
inputs/0????????? 
&?#
inputs/1?????????
p

 
? "???????????
&__inference_model_layer_call_fn_679897?,.-"#f?c
\?Y
O?L
"?
inputs/0????????? 
&?#
inputs/1?????????
p 

 
? "???????????
A__inference_re_lu_layer_call_and_return_conditional_losses_680988X/?,
%?"
 ?
inputs?????????(
? "%?"
?
0?????????(
? u
&__inference_re_lu_layer_call_fn_680993K/?,
%?"
 ?
inputs?????????(
? "??????????(?
C__inference_reshape_layer_call_and_return_conditional_losses_681023\/?,
%?"
 ?
inputs?????????
? ")?&
?
0?????????
? {
(__inference_reshape_layer_call_fn_681028O/?,
%?"
 ?
inputs?????????
? "???????????
$__inference_signature_wrapper_679309?,.-"#m?j
? 
c?`
0
input_1%?"
input_1?????????
,
input_2!?
input_2????????? "5?2
0
reshape%?"
reshape?????????