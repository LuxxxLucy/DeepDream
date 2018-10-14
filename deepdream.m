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



decoder = NetDecoder[{"Image"}];
l2Loss = NetGraph[{MeanSquaredLossLayer[]}, {1 -> NetPort["Objective"]}];

DeepDreamStep[contentImg_, featureNet_, iterStep_] :=

 Module[{dims, trainingdata, trainStep, diff, absDiff},
  dims = Prepend[3]@Reverse@ImageDimensions[contentImg];
  net = NetGraph[
    <|"Image" ->
      ConstantArrayLayer[
       "Array" ->
        NetEncoder[{"Image", ImageDimensions[contentImg]}]@
         contentImg],
     "imageFeat" -> NetReplacePart[featureNet, "Input" -> dims],
     "l2Loss" -> l2Loss|>,
    {
     "Image" -> "imageFeat",
     {"imageFeat", NetPort["ZeroBaseTensor"]} -> "l2Loss"
     }
    ];
  trainingdata = <|
    "ZeroBaseTensor" -> {
      NetReplacePart[featureNet, "Input" -> dims][
        NetEncoder[{"Image", ImageDimensions[contentImg]}]@
         contentImg]*0}|>;
  trainStep = NetTrain[net,
    trainingdata,
    LossFunction -> {"Objective" -> Scaled[-1]},
    LearningRateMultipliers -> {"Image" -> 1, _ -> None},
    TrainingProgressReporting ->
     Function[decoder[#Weights[{"Image", "Array"}]]],
    (*TrainingProgressReporting\[Rule] None,*)
    BatchSize -> 1,

    MaxTrainingRounds -> iterStep,
    Method -> {"SGD", "LearningRate" -> 1},
    TargetDevice -> "CPU"];
  diff = ShowResult[trainStep];
  absDiff = Nest[ Mean, ImageDifference[diff, contentImg], 2];
  ImageSubtract[diff, contentImg]/absDiff*0.015
  ]

  Octave[contentImg_, featureNet_, iterStep_, octave_, octaveScale_,
  jitter_] :=
 Module[
  {jitterTmp, diffImg, changedImg, jitterImg},
  If[octave <= 1,
   {
    jitterTmp = RandomInteger[{-jitter, jitter + 1}, 2];
    jitterImg =
     ImageTransformation[ contentImg, # + jitterTmp &,
      DataRange -> Full];
    diffImg = DeepDreamStep[jitterImg, featureNet, iterStep];
    diffImg =
     ImageTransformation[ diffImg,  # - jitterTmp &,
      DataRange -> Full];
    diffImg
    },
   {
    diffImg =
     Octave[ImageResize[contentImg, Scaled[1/octaveScale]],
      featureNet, iterStep, octave - 1, octaveScale, jitter];
    diffImg = ImageResize[diffImg, Scaled[octaveScale]];
    currentDiff = DeepDreamStep[contentImg, featureNet, iterStep];

    diffImg = ImageAdd[diffImg, currentDiff]
    }
   ];
  diffImg
  ]

OctaveStep[dreamSeed_, featureNet_, iterStep_, octave_, octaveScale_,
  jitter_] :=

 ImageAdd[
  Octave[dreamSeed, featureNet, iterStep, octave, octaveScale,
   jitter], dreamSeed]


DeepDreamMaker[dreamSeed_, featureNet_, iterStep_, octave_,
  octaveScale_, jitter_] :=
 Module[{result},
  result = dreamSeed;
  Do[result =
    OctaveStep[result, featureNet, iterStep, octave, octaveScale,
     jitter]
   , {1}
   ];
  result
  ]
