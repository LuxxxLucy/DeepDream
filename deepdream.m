(* ::Package:: *)
(*
    Deep Dream Algorithm. (2015 Google)

    Implements:
        1. Deep Dream maxing l2-norm. gradient updating step.
        2. Octave (multiplie scaling)

    TODO:
        1. iterstep control. also leave out the learning rate.(decay?)
        2. with or without guide ( the idea is in Style transfer)
        3. Test with different filters

    by Jialin Lu https://luxxxlucy.github.io
*)

DeepDreamStep[contentImg_, featureNet_] :=

 Module[{dims, net, trainingdata, trainStep},
  dims = Prepend[3]@Reverse@ImageDimensions[contentImg];
  net = NetGraph[
    <|"Image" ->
      ConstantArrayLayer[
       "Array" ->
        NetEncoder[{"Image", ImageDimensions[contentImg]}]@ contentImg],
     "imageFeat" -> NetReplacePart[featureNet, "Input" -> dims],
     "l2Loss" -> l2Loss
     |>,
    {
     "Image" -> "imageFeat",
     {"imageFeat", NetPort["ZeroBaseTensor"]} -> "l2Loss" }
    ];
  trainingdata = <|
    "ZeroBaseTensor" ->
    { NetReplacePart[featureNet, "Input" -> dims][ NetEncoder[{"Image", ImageDimensions[contentImg]}] @ contentImg]*0}|>;
  trainStep = NetTrain[net,
    trainingdata,
    LossFunction -> {"Objective" -> Scaled[-1]},
    LearningRateMultipliers -> {"Image" -> 1, _ -> None},
    (* TrainingProgressReporting ->
     Function[decoder[#Weights[{"Image", "Array"}]]], *)
    TrainingProgressReporting\[Rule] None,
    MaxTrainingRounds -> 300,
    Method -> {"SGD", "LearningRate" -> 25},
    TargetDevice -> "CPU"];
  ShowResult[trainStep]
  ]

(*
    multi-scale process by so-called Octave.

    A recursive way to do it.

    Note that the Octave function do not return the modified image, but instead, the
*)

Octave[contentImg_, featureNet_, octave_, octaveScale_, jitter_] :=

 Module[
  {jitterTmp, diffImg, changedImg, jitterImg},
  If[octave <= 1,
   {
    (* every time offset the image by an random jitter. *)
    jitterTmp = RandomInteger[{-jitter, jitter + 1}, 2];
    jitterImg =
     ImageTransformation[ contentImg, # + jitterTmp &,
      DataRange -> Full];
    diffImg =
     ImageSubtract[DeepDreamStep[jitterImg, featureNet], jitterImg];
    (* and jitter back *)
    diffImg =
     ImageTransformation[ diffImg,  # - jitterTmp &,
      DataRange -> Full];
    },
   {
    diffImg =
     Octave[ImageResize[contentImg, Scaled[1/octaveScale]],
      featureNet, octave - 1, octaveScale, jitter];
    temp =
     ImageAdd[ImageResize[diffImg, Scaled[octaveScale]],
      DeepDreamStep[contentImg, featureNet] ];
    diffImg = ImageSubtract[temp, contentImg]
    }
   ];
  diffImg
  ]
