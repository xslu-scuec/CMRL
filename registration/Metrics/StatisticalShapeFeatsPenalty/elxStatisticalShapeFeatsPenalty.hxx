/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

#ifndef __elxStatisticalShapeFeatsPenalty_HXX__
#define __elxStatisticalShapeFeatsPenalty_HXX__

#include "elxStatisticalShapeFeatsPenalty.h"

#include "itkPointSet.h"
#include "itkDefaultStaticMeshTraits.h"
#include "itkVTKPolyDataReader.h"
#include "itkVTKPolyDataWriter.h"
#include <itkMesh.h>

#include <typeinfo>


namespace elastix
{

/**
 * ******************* Initialize ***********************
 */

template <class TElastix>
void
StatisticalShapeFeatsPenalty<TElastix>
::Initialize( void ) throw (itk::ExceptionObject)
{
  itk::TimeProbe timer;
  timer.Start();
  this->Superclass1::Initialize();
  timer.Stop();
  elxout << "Initialization of StatisticalShapeFeatsPenalty metric took: "
	  << static_cast<long>( timer.GetMean() * 1000 ) << " ms." << std::endl;

} // end Initialize()


/**
 * ***************** BeforeRegistration ***********************
 */

template <class TElastix>
void StatisticalShapeFeatsPenalty<TElastix>
::BeforeRegistration( void )
{
  /** Read and set the fixed pointset. */
  std::string fixedName = this->GetConfiguration()->GetCommandLineArgument( "-fp" );
  typename PointSetType::Pointer fixedPointSet      = 0;
  const typename ImageType::ConstPointer fixedImage = this->GetElastix()->GetFixedImage();
  const unsigned int nrOfFixedPoints = this->ReadShape(
    fixedName, fixedPointSet, fixedImage );
  this->SetFixedPointSet( fixedPointSet );

  /** Read and set the meanVector filename. */
  std::string                  meanVectorName = this->GetConfiguration()->GetCommandLineArgument( "-meanf" );
  vcl_ifstream                 datafile;
  vnl_vector< double > * const meanVector = new vnl_vector< double >();
  datafile.open( meanVectorName.c_str() );
  if( datafile.is_open() )
  {
    meanVector->read_ascii( datafile );
    datafile.close();
    datafile.clear();
    elxout << " meanVector " << meanVectorName << " read" << std::endl;
  }
  else
  {
    itkExceptionMacro( << "Unable to open meanVector file: " << meanVectorName );
  }
  this->SetMeanVector( meanVector );

  /** Read and set the covariance matrix filename. */
  std::string covarianceMatrixName = this->GetConfiguration()->GetCommandLineArgument( "-covariancef" );

  vnl_matrix< double > * const covarianceMatrix = new vnl_matrix< double >();

  datafile.open( covarianceMatrixName.c_str() );
  if( datafile.is_open() )
  {
    covarianceMatrix->read_ascii( datafile );
    datafile.close();
    datafile.clear();
    elxout << "covarianceMatrix " << covarianceMatrixName << " read" << std::endl;
  }
  else
  {
    itkExceptionMacro( << "Unable to open covarianceMatrix file: " << covarianceMatrixName );
  }
  this->SetCovarianceMatrix( covarianceMatrix );

} // end BeforeRegistration()


/**
 * ***************** BeforeEachResolution ***********************
 */

template <class TElastix>
void StatisticalShapeFeatsPenalty<TElastix>
::BeforeEachResolution( void )
{
  /** Get the current resolution level. */
  unsigned int level
    = this->m_Registration->GetAsITKBaseType()->GetCurrentLevel();

  /** Get and set ShrinkageIntensityFeats. Default 0.2. */
  double shrinkageIntensityFeats = 0.2;
  this->GetConfiguration()->ReadParameter( shrinkageIntensityFeats, "ShrinkageIntensityFeats",
    this->GetComponentLabel(), level, 0 );
  this->SetShrinkageIntensityFeats( shrinkageIntensityFeats );
  
  /** Get and set BaseVarianceFeats. Default 10.0. */
  double baseVarianceFeats = 10.0;
  this->GetConfiguration()->ReadParameter( baseVarianceFeats,
    "BaseVarianceFeats", this->GetComponentLabel(), level, 0 );
  this->SetBaseVarianceFeats( baseVarianceFeats );

} // end BeforeEachResolution()


/**
 * ************** TransformPointsSomePointsVTK *********************
 *
 * This function reads points from a .vtk file and transforms
 * these fixed-image coordinates to moving-image
 * coordinates.
 *
 * Reads the inputmesh from a vtk file, assuming world coordinates.
 * Computes the transformed points, save as outputpoints.vtk.
 */

template<class TElastix>
unsigned int
StatisticalShapeFeatsPenalty< TElastix >
::ReadShape(
  const std::string & ShapeFileName,
  typename PointSetType::Pointer & pointSet,
  const typename ImageType::ConstPointer image )
{
  /** Typedef's. \todo test DummyIPPPixelType=bool. */
  typedef double DummyIPPPixelType;
  typedef DefaultStaticMeshTraits<
    DummyIPPPixelType, FixedImageDimension,
    FixedImageDimension, CoordRepType >                  MeshTraitsType;
  typedef Mesh< DummyIPPPixelType,
    FixedImageDimension, MeshTraitsType >                MeshType;
  typedef VTKPolyDataReader< MeshType >                  MeshReaderType;

  /** Read the input points. */
  typename MeshReaderType::Pointer meshReader = MeshReaderType::New();
  meshReader->SetFileName( ShapeFileName.c_str() );
  elxout << "  Reading input point file: " << ShapeFileName << std::endl;
  try
  {
    meshReader->Update();
  }
  catch( ExceptionObject & err )
  {
    xl::xout[ "error" ] << "  Error while opening input point file." << std::endl;
    xl::xout[ "error" ] << err << std::endl;
  }

  /** Some user-feedback. */
  elxout << "  Input points are specified in world coordinates." << std::endl;
  unsigned long nrofpoints = meshReader->GetOutput()->GetNumberOfPoints();
  elxout << "  Number of specified input points: " << nrofpoints << std::endl;

  typename MeshType::Pointer mesh = meshReader->GetOutput();
  pointSet                        = PointSetType::New();
  pointSet->SetPoints( mesh->GetPoints() );
  return nrofpoints;

} // end ReadShape()


} // end namespace elastix


#endif // end #ifndef __elxStatisticalShapeFeatsPenalty_HXX__


