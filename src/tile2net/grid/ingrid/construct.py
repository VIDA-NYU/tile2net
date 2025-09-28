from ..frame.namespace import namespace
import pooch

from .ingrid import InGrid


class VRAM24(
    namespace
):
    def from_boston_common(
            self,
            outdir='./outdir',
            location='Boston Common',
            pad=2,
            length=8,
    ) -> InGrid:
        """Small park in downtown Boston."""
        return InGrid.from_basic(
            outdir=outdir,
            location=location,
            pad=pad,
            length=length,
        )

    def from_manhattan(
            self,
            outdir='./outdir',
            location="40.7053, -74.0111, 40.7119, -73.9900",
            pad=2,
            length=8,
    ) -> InGrid:
        """Medium sized area in downtown Manhattan."""
        return InGrid.from_basic(
            outdir=outdir,
            location=location,
            pad=pad,
            length=length,
        )

    def from_frisco(
            self,
            outdir='./outdir',
            location="37.7872, -122.4017, 37.7939, -122.3873",
            pad=1,
            length=8,
    ) -> InGrid:
        """Downtown San Francisco (Frisco) area."""
        return InGrid.from_basic(
            outdir=outdir,
            location=location,
            pad=pad,
            length=length,
        )

    def from_berkeley(
            self,
            outdir='./outdir',
            location="37.8684, -122.2658, 37.8809, -122.2441",
            pad=2,
            length=8,
    ) -> InGrid:
        """Downtown Berkeley area near the university."""
        return InGrid.from_basic(
            outdir=outdir,
            location=location,
            pad=pad,
            length=length,
        )

    def from_augusta(
            self,
            outdir='./outdir',
            location="44.3112, -69.7708, 44.3185, -69.7505",
            pad=2,
            length=8,
    ) -> InGrid:
        """Downtown Augusta, Maine area."""
        return InGrid.from_basic(
            outdir=outdir,
            location=location,
            pad=pad,
            length=length,
        )

    def from_spring_hill(
            self,
            outdir='./outdir',
            location="35.7526, -86.9171, 35.7664, -86.8819",
            pad=2,
            length=8,
    ) -> InGrid:
        """Spring Hill, Tennessee area."""
        return InGrid.from_basic(
            outdir=outdir,
            location=location,
            pad=pad,
            length=length,
        )



class VRAM8(
    namespace
):
    def from_boston_common(
            self,
            outdir='./outdir',
            location='Boston Common',
            pad=1,
            length=8,
    ) -> InGrid:
        return InGrid.from_basic(
            outdir=outdir,
            location=location,
            pad=pad,
            length=length,
        )


class Construct(
    namespace
):
    @VRAM8
    def vram8(self):
        """Constructors optimized for local inference with 8 GB VRAM."""

    @VRAM24
    def vram24(self):
        """Constructors optimized for local inference with 24 GB VRAM."""
